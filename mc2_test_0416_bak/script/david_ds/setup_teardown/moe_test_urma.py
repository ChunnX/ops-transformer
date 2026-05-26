import os
import socket
from typing import List
import torch
import gc
import torch_npu
import torchair as tng
import csv
import numpy as np
from torch.multiprocessing import Process, Manager
import torch.distributed as dist
from quantize import torch_to_numpy_tensor, numpy_to_torch_tensor, get_dtype_int_value
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import toolss
import time
import pandas as pd
from enum import Enum
from math import ceil
import sys
torch.set_printoptions(profile="full")
import random
import queue


# 控制模式
is_dispatch = 0
is_combine = 0
is_continue = 1
is_performance = 0  # 是否跑性能模式
is_topk_balance = False  # 是否均衡topK
is_x_random = True  # 是否让x值随机

# loop次数
loop = 105          # 性能loop
loop_acc = 1      # 精度loop

moe_expert_num = 32  # moe专家数
bs = 256
h = 7168
k = 8
x_dtype = 'bf16'
y_dtype = 'fp8_e5m2'
expandx_dtype = 'fp8_e5m2'
quant_mode = 4
random_seed = 8


graph_type = 0  # 0为单算子，1为静态图模式，2为动态图模式, 3为aclGraph
rank_per_dev = 2  # 每个host有几个die
comm_quant_mode = 0
is_quant_scales = False
is_continue_golden_expand_x = True  # 是否在连跑模式下，将dispatch的golden值直接传给combine算子
is_full_core = True  # True时使用全部的核，False时使用一半的核
core_num = 16 if not is_full_core else 32                       # 非全核模式使用的核数（单个核上有一个aic核，一个aiv核）
aic_core_num = core_num                                         # 非全核模式使用的aic核数
aiv_core_num = core_num * 2                                     # 非全核模式使用的aiv核数
is_mask = 0  # 是否使能active_mask
is_variable_bs = 0  # 是否使能可变bs
is_variable_local_moe_expert_num = 0                           # 是否使能可变moe卡上专家数
out_dtype = 1 if x_dtype == 'fp16' else 0  # 控制dispatch输入及combine输出的数据类型， 0为bfloat16，1为float16
server_num = 1  # 单机或者多机模式
is_cloud = False  # 是否在云上环境进行测试，使用云上环境时设置为True，使用物理机时设为False
tp_world_size = 1
shared_expert_num = 0  # 共享专家数
rank_num_per_shared_expert = 1  # 每个共享专家占据的卡数
is_shared_expert_x = 0

comm_type = 2  # 2为AIV直驱URMA通信
comm_alg =''

if not is_cloud:
    server_index = 0  # 多机每个脚本保持一致并递增index值
    master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'  # 如果通信不通，查看一下master_ip是否设对
else:
    # 云上环境用下面方式获取server_index和master_ip
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

input_dtype = torch.bfloat16 if out_dtype == 0 else torch.float16  # dispatch输入及combine输出的数据类型
combine_input_dtype = input_dtype

world_size = server_num * rank_per_dev  # 总die数
server_ranks = slice(server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev)  # 本机各卡索引


is_unfixed_real_token = is_mask or is_variable_bs  # 每张卡上真实token数是否固定

shared_expert_rank_num = shared_expert_num * rank_num_per_shared_expert
ep_world_size = world_size
moe_rank_num = ep_world_size - shared_expert_rank_num
local_moe_expert_num = 16
global_bs = bs * ep_world_size
max_shared_group_num = ceil(ep_world_size / rank_num_per_shared_expert) if (rank_num_per_shared_expert > 0) else 0
A_shared = bs * max_shared_group_num
A_moe = (global_bs * min(local_moe_expert_num, k)) if global_bs != 0 else (bs * ep_world_size * min(local_moe_expert_num, k))
is_shared = (shared_expert_rank_num > 0)
card_rate = ((ep_world_size - shared_expert_rank_num) // shared_expert_rank_num) if is_shared else 0
is_quant = (quant_mode > 0)
is_save_tensor_to_bin = False

# 控制 dispatch 输出的 expert_token_nums 的格式：
# 0 表示每个专家收到 token 数量的前缀和；
# 1 表示每个专家收到的 token 数量（默认）
expert_token_nums_type = 1

class ComparedOutput(Enum):
    Y_OUT = "y_out"
    EP_ASSIST_INFO = "assist_info_for_combine"
    EXPAND_X = "expand_x"
    EXPAND_IDX = "expand_idx"
    DYNAMIC_SCALES = "dynamic_scales"
    EXPERT_TOKEN_NUMS = "expert_token_nums"
    X = "x"


os.environ['HCCL_BUFFSIZE'] = '5000'
current_case_name = 'moe_ds_network_2p_ID023'

# region 常用常量
# region 性能处理相关常量
dispatch_profiling_path = './Profiling_Result_mc2_dispatch_16p'
combine_profiling_path = './Profiling_Result_mc2_combine_16p'
continue_profiling_path = './Profiling_Result_mc2_both_16p'
dispatch_setup_op_type = 'MoeDistributeDispatchSetup'
dispatch_teardown_op_type = 'MoeDistributeDispatchTeardown'
combine_setup_op_type = 'MoeDistributeCombineSetup'
combine_teardown_op_type = 'MoeDistributeCombineTeardown'
path_to_op_types = {
    dispatch_profiling_path: [dispatch_setup_op_type, dispatch_teardown_op_type],
    combine_profiling_path: [combine_setup_op_type, combine_teardown_op_type],
    continue_profiling_path: [dispatch_setup_op_type, dispatch_teardown_op_type, combine_setup_op_type, combine_teardown_op_type]
}

path_to_Allocated_type = {
    dispatch_profiling_path: ['npu::npu_moe_distribute_dispatch_setup', 'npu::npu_moe_distribute_dispatch_teardown'],
    combine_profiling_path: ['npu::npu_moe_distribute_combine_setup', 'npu::npu_moe_distribute_combine_teardown'],
    continue_profiling_path: ['npu::npu_moe_distribute_dispatch_setup', 'npu::npu_moe_distribute_dispatch_teardown', 'npu::npu_moe_distribute_combine_setup', 'npu::npu_moe_distribute_combine_teardown'],
}

# endregion

# region 通信域相关常量
ep_ranks_list = [list(range(0, world_size))]


# endregion
# endregion

# region 工具函数
# region 张量处理函数
def chunk_tensor(tensor: torch.Tensor, chunks: int) -> List[torch.Tensor]:
    return list(tensor.chunk(chunks))


def split_tensor(tensor: torch.Tensor, split_size_or_sections: int or List[int]) -> List[torch.Tensor]:
    return list(tensor.split(split_size_or_sections))


def divide_per_card(tensor_by_expert, card_num, num_per_expert):
    '''将以专家为单位的张量按卡切分'''

    if card_num == world_size:
        # 全域
        num_per_card_list = (
                [1 * num_per_expert] * shared_expert_rank_num
                + [local_moe_expert_num * num_per_expert] * moe_rank_num
        )
    elif card_num == ep_world_size:
        # ep域
        num_per_card_list = (
                [1 * num_per_expert] * shared_expert_rank_num
                + [local_moe_expert_num * num_per_expert] * moe_rank_num
        )
    else:
        print("divide_per_card only support world_size and ep_world_size")
        exit(0)

    return split_tensor(tensor_by_expert, num_per_card_list)


# endregion

# region 测试执行前预备工作相关函数

def clean_previous_profiling_result():
    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')

def print_setting():
    print(f"{loop_acc=}")
    print(f"{server_num=}")
    print(f"{ep_world_size=}")
    print(f"{shared_expert_rank_num=}")
    print(f"{moe_expert_num=}")
    print(f"{local_moe_expert_num=}")
    print(f"{bs=}")
    print(f"{h=}")
    print(f"{k=}")
    print(f"{quant_mode=}")
    print(f"{comm_quant_mode=}")
    print(f"{shared_expert_num=}")
    print(f"{shared_expert_rank_num=}")
    print(f"{rank_num_per_shared_expert=}")
    print(f"x_dtype={x_dtype}")
    print(f"y_dtype={y_dtype}")
    print(f"expandx_dtype={expandx_dtype}")
    print(f"combine_input_dtype={combine_input_dtype}")
    print(f"{random_seed=}")

def check_setting():
    
    if shared_expert_rank_num > ep_world_size:
        print("shared_expert_rank_num 不能大于 ep_world_size")
        exit(0)

    # if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
    #     print("存在共享专家时, ep_world_size 必须是 sharedExpertRankNum的整数倍")
    #     exit(0)

    if moe_expert_num % moe_rank_num != 0:
        print("moe_expert_num 必须是 moe_rank_num 的整数倍")
        exit(0)

    if quant_mode == 1:
        print("当前不支持静态量化，quant_mode合法值仅有 0 和 2")
        exit(0)

    if is_mask and is_variable_bs:
        print("掩膜 与 可变bs 特性不能同时使能")
        exit(0)

    if comm_quant_mode == 1:
        print("当前不支持 int12 通信量化, comm_quant_mode 合法值仅有 0 和 2")
        exit(0)

    # if is_continue_golden_expand_x == False:
    #     print("当前仅支持 is_continue_golden_expand_x = True")
    #     exit(0)

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
def get_profiling(loop, path):
    repeat = 1
    warmup = 5
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
        experimental_config=experimental_config, )

    return prof


# endregion
# endregion

# region 性能数据处理函数
# def get_performance(profiling_path, name):
def get_execution_time(profiling_path: str):
    op_types = path_to_op_types[profiling_path]
    rank_types = ['shared', 'moe']
    performance_info = {(op_type, rank_type): [] for op_type in op_types for rank_type in rank_types}

    long_wait_ranks = set()
    # print("291path is ",profiling_path)
    for dirpath, _, filenames in os.walk(profiling_path):
        # print("filenames!",filenames)
        for filename in filenames:
            # print("enter294!")
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                local_rank_id = op_summary['Device_id'].iloc[0]
                ep_rank_id = (rank_per_dev * server_index + local_rank_id)
                rank_type = 'shared' if ep_rank_id < shared_expert_rank_num else 'moe'
                for op_type in op_types:
                    op_info = op_summary[op_summary['OP Type'] == op_type]

                    op_execute_time_series = op_info['Task Duration(us)'].tail(50)
                    performance_info[(op_type, rank_type)].append(op_execute_time_series)

                    mean_op_wait_time = op_info['Task Wait Time(us)'].tail(50).mean()
                    if mean_op_wait_time > 5:
                        long_wait_ranks.add(local_rank_id)

    print(f"Task Wait > 5 us local rank id is {sorted(long_wait_ranks)}")
    print("performance_info:",performance_info)
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
    # print("print_dispatch_performance320:",mean_performances + min_performances)
    # 顺序：平均，最小；dispatch, combine; 共享，moe
    return mean_performances + min_performances


def get_memory_usage(profiling_path: str, name: str):
    columns = ['rank_id', 'memory_usage']
    memory_usage_per_rank = pd.DataFrame(columns=columns)
    

    if graph_type == 1:
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

    
    else:
        for dirpath, _, filenames in os.walk(profiling_path):
            for filename in filenames:
                if filename.startswith("operator_memory"):
                    memory_detail = pd.read_csv(os.path.join(dirpath, filename))

                    local_rank_id = int(memory_detail['Device Type'][0].split(":")[1])  # memory_record['Device Type']值形如 "NPU:6"
                    rank_id = rank_per_dev * server_index + local_rank_id

                    
                    memory_info = memory_detail[memory_detail['Name'] == name]

                    rank_Allocation_Total_Allocated = memory_info['Allocation Total Allocated(MB)']
                    rank_Release_Total_Allocated = memory_info['Release Total Allocated(MB)']
                    rank_Release_Total_Active = memory_info['Release Total Active(MB)']

                    rank_max_allocated_memory = max(rank_Allocation_Total_Allocated.max(), rank_Release_Total_Allocated.max(), rank_Release_Total_Active.max())
                    rank_min_allocated_memory = min(rank_Allocation_Total_Allocated.min(), rank_Release_Total_Allocated.min(), rank_Release_Total_Active.min())
                    rank_memory_usage = rank_max_allocated_memory - rank_min_allocated_memory

                    memory_usage_per_rank.loc[memory_usage_per_rank.shape[0]] = [rank_id, rank_memory_usage]

    memory_usage_per_rank['rank_id'] = memory_usage_per_rank['rank_id'].astype('int32')
    memory_usage_per_rank.sort_values(by='rank_id', inplace=True)
    memory_usage_per_rank.reset_index(drop=True, inplace=True)
    return memory_usage_per_rank


def print_dataframe(df: pd.DataFrame):
    print('\n' + '-' * 60)
    print(df.to_string(index=False))
    print('-' * 60 + '\n')

def print_dispatch_performance(dispatch_performance):

    (
        dispatch_setup_share_performance, dispatch_setup_moe_performance, dispatch_teardown_share_performance, dispatch_teardown_moe_performance,
        min_dispatch_setup_share_performance, min_dispatch_setup_moe_performance, min_dispatch_teardown_share_performance, min_dispatch_teardown_moe_performance, 
    ) = dispatch_performance
    print("print_dispatch_performance:",dispatch_performance)
    print(
        f'[INFO] MC2融合算子Average性能为:'
        f'dispatch_setup共享: {dispatch_setup_share_performance} us, dispatch_setup Moe: {dispatch_setup_moe_performance} us, '
        f'dispatch_teardown共享: {dispatch_teardown_share_performance} us, dispatch_teardown Moe: {dispatch_teardown_moe_performance} us\n'
    )
    print(
        f'[INFO] MC2融合算子Min性能为:'
        f'dispatch_setup共享:{min_dispatch_setup_share_performance} us, dispatch_setup Moe:{min_dispatch_setup_moe_performance} us, '
        f'dispatch_teardown共享:{min_dispatch_teardown_share_performance} us, dispatch_teardown Moe:{min_dispatch_teardown_moe_performance} us\n'
    )


def print_dispatch_performance_memory(dispatch_performance):
    # print(f'[INFO] Dispatch算子内存占用:')
    print_dataframe(dispatch_performance)


def print_combine_performance(combine_performance):

    (
        combine_setup_share_performance, combine_setup_moe_performance, combine_teardown_share_performance, combine_teardown_moe_performance, 
        min_combine_setup_share_performance, min_combine_setup_moe_performance, min_combine_teardown_share_performance, min_combine_teardown_moe_performance, 
    ) = combine_performance

    print(
        f'[INFO] MC2融合算子Average性能为:'
        f'combine_setup共享: {combine_setup_share_performance} us, combine_setup Moe: {combine_setup_moe_performance} us, '
        f'combine_teardown共享: {combine_teardown_share_performance} us, combine_teardown Moe: {combine_teardown_moe_performance} us\n'
    )
    print(
        f'[INFO] MC2融合算子Min性能为:'
        f'combine_setup共享:{min_combine_setup_share_performance} us, combine_setup Moe:{min_combine_setup_moe_performance} us, '
        f'combine_teardown共享:{min_combine_teardown_share_performance} us, combine_teardown Moe:{min_combine_teardown_moe_performance} us\n'
    )
    # print(f'[INFO] Combine算子内存占用:')
    # print_dataframe(memory_usage)


def print_combine_performance_memory(combine_performance):
    # print(f'[INFO] Combine算子内存占用:')
    print_dataframe(combine_performance)


def print_continue_performance(continue_performance):
    (
    dispatch_setup_share_performance, dispatch_setup_moe_performance, dispatch_teardown_share_performance, dispatch_teardown_moe_performance, 
    combine_setup_share_performance, combine_setup_moe_performance, combine_teardown_share_performance, combine_teardown_moe_performance, 
    min_dispatch_setup_share_performance, min_dispatch_setup_moe_performance, min_dispatch_teardown_share_performance, min_dispatch_teardown_moe_performance, 
    min_combine_setup_share_performance, min_combine_setup_moe_performance, min_combine_teardown_share_performance, min_combine_teardown_moe_performance,
    ) = continue_performance

    print(
        f'[INFO] MC2融合算子Average性能为:'
        f'dispatch_setup共享: {dispatch_setup_share_performance} us, dispatch_setup Moe: {dispatch_setup_moe_performance} us, '
        f'dispatch_teardown共享: {dispatch_teardown_share_performance} us, dispatch_teardown Moe: {dispatch_teardown_moe_performance} us\n'
        f'combine_setup共享: {combine_setup_share_performance} us, combine_setup Moe: {combine_setup_moe_performance} us, '
        f'combine_teardown共享: {combine_teardown_share_performance} us, combine_teardown Moe: {combine_teardown_moe_performance} us\n'
    )
    print(
        f'[INFO] MC2融合算子Min性能为:'
        f'dispatch_setup共享:{min_dispatch_setup_share_performance} us, dispatch_setup Moe:{min_dispatch_setup_moe_performance} us, '
        f'dispatch_teardown共享:{min_dispatch_teardown_share_performance} us, dispatch_teardown Moe:{min_dispatch_teardown_moe_performance} us\n'
        f'combine_setup共享:{min_combine_setup_share_performance} us, combine_setup Moe:{min_combine_setup_moe_performance} us, '
        f'combine_teardown共享:{min_combine_teardown_share_performance} us, combine_teardown Moe:{min_combine_teardown_moe_performance} us\n'
    )
    # print(f'[INFO] Dispatch与Combine算子总内存占用:')
    # print_dataframe(memory_usage)


def print_continue_performance_memory(continue_performance):
    # print(f'[INFO] Dispatch算子内存占用:')
    print_dataframe(continue_performance)


def get_dispatch_performance():
    dispatch_performance = get_execution_time(dispatch_profiling_path)
    # print("print_dispatch_performance462:",dispatch_performance)
    dispatch_performace_memory_list = []

    if graph_type == 1:
        dispatch_performace_memory = get_memory_usage(dispatch_profiling_path, None)
        dispatch_performace_memory_list.append(dispatch_performace_memory)
    
    else:
        op_types = path_to_Allocated_type[dispatch_profiling_path]
        for op_type in op_types:
            dispatch_performace_memory = get_memory_usage(dispatch_profiling_path, op_type)
            dispatch_performace_memory_list.append(dispatch_performace_memory)

    return dispatch_performance, dispatch_performace_memory_list


def get_combine_performance():
    combine_performance = get_execution_time(combine_profiling_path)
    combine_performance_memory_list = []

    if graph_type == 1:
        combine_performance_memory = get_memory_usage(combine_profiling_path, None)
        combine_performance_memory_list.append(combine_performance_memory)

    else:
        op_types = path_to_Allocated_type[combine_profiling_path]
        for op_type in op_types:
            combine_performance_memory = get_memory_usage(combine_profiling_path, op_type)
            combine_performance_memory_list.append(combine_performance_memory)

    return combine_performance, combine_performance_memory_list


def get_continue_performance():
    continue_performance = get_execution_time(continue_profiling_path)
    continue_performance_memory_list = []

    if graph_type == 1:
        continue_performance_memory = get_memory_usage(continue_profiling_path, None)
        continue_performance_memory_list.append(continue_performance_memory)



    else:
        op_types = path_to_Allocated_type[continue_profiling_path]
        for op_type in op_types:
            continue_performance_memory = get_memory_usage(continue_profiling_path, op_type)
            continue_performance_memory_list.append(continue_performance_memory)

    return continue_performance, continue_performance_memory_list


# endregion

# region 图模式相关类和函数
def compile_model(model):
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(0)

    print(f"------------------graph_type: {graph_type}------------------", )
    import torchair

    from torchair.configs.compiler_config import CompilerConfig

    # 配置使用的核数
    compiler_config = CompilerConfig()
    if not is_full_core:
        print(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
    compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
    

    if graph_type == 1:
        # 传统入图模式，静态shape+在线编译场景
        print('graph_type=1, 测试场景: 传统入图模式，静态shape+在线编译场景')
        npu_backend = torchair.get_npu_backend(compiler_config= None if is_full_core else compiler_config)
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:
        # ACLNN入图模式，动态shape+二进制
        print('graph_type=2, 测试场景: ACLNN入图模式，动态shape+二进制场景')
        npu_backend = torchair.get_npu_backend(compiler_config= None if is_full_core else compiler_config)
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)
    elif graph_type == 3:
        # 测试aclGraph
        print('graph_type=3, 测试场景: aclGraph')
        compiler_config.mode = "reduce-overhead"
        compiler_config.debug.aclgraph.clone_input = False
        compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)

    return compiled_model


class MOE_DISTRIBUTE_DISPATCH_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_ids, x_active_mask, group_ep, ep_world_size,
                ep_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, moe_expert_num,
                scales, quant_mode, global_bs, comm_type, expert_token_nums_type, comm_alg):
        (
        y, expand_idx, comm_cmd_info
        ) = torch_npu.npu_moe_distribute_dispatch_setup(
        x=x,
        expert_ids=expert_ids,
        group_ep=group_ep,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        scales=scales,
        x_active_mask=x_active_mask,
        expert_shard_type=expert_shard_type,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        comm_alg=comm_alg,
        )


        (
        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
        ) = torch_npu.npu_moe_distribute_dispatch_teardown(
        x=x,
        y=y,
        expert_ids=expert_ids,
        comm_cmd_info=comm_cmd_info,
        group_ep=group_ep,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        expert_shard_type=expert_shard_type,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        quant_mode=quant_mode,
        global_bs=global_bs,
        expert_token_nums_type=1,
        comm_type=comm_type,
        comm_alg=comm_alg,
        )

        return expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, expand_idx

class MOE_DISTRIBUTE_COMBINE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, expand_x, expert_ids, expand_idx, assist_info_for_combine, x_active_mask, expert_scales,
            group_ep, shared_expert_x,shared_expert_num,
            ep_world_size, ep_rank_id, expert_shard_type, shared_expert_rank_num,
            moe_expert_num, comm_quant_mode, global_bs, comm_type, comm_alg,
    ):

        quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(
            expand_x=expand_x,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            group_ep=group_ep,
            ep_world_size = ep_world_size,
            ep_rank_id = ep_rank_id,
            moe_expert_num=moe_expert_num,
            expert_shard_type=0,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            global_bs=global_bs,
            comm_quant_mode=comm_quant_mode,
            comm_type=comm_type,
            comm_alg=comm_alg,

        )


        output = torch_npu.npu_moe_distribute_combine_teardown(
            expand_x=expand_x,
            quant_expand_x=quant_expand_x,
            expert_ids=expert_ids,
            expand_idx=expand_idx,
            expert_scales=expert_scales,
            comm_cmd_info=comm_cmd_info,
            group_ep=group_ep,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            x_active_mask=x_active_mask,
            shared_expert_x=None,
            expert_shard_type=0,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            global_bs=global_bs,
            comm_quant_mode=comm_quant_mode,
            comm_type=comm_type,
            comm_alg=comm_alg,
        )


        return output


class MOE_DISTRIBUTE_CONTINUE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, x, expert_ids, x_active_mask, group_ep, ep_world_size, shared_expert_num,expand_idx,
            ep_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num, expand_x,assist_info_for_combine,
            shared_expert_x,expert_token_nums_type, 
            scales, quant_mode, comm_quant_mode, global_bs, expert_scales, golden_expand_x, comm_type, comm_alg
    ):
         
        (
            y, expand_idx, comm_cmd_info
        ) = torch_npu.npu_moe_distribute_dispatch_setup(
            x=x,
            expert_ids=expert_ids,
            group_ep=group_ep,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            scales=scales,
            x_active_mask=x_active_mask,
            expert_shard_type=expert_shard_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            quant_mode=quant_mode,
            global_bs=global_bs,
            comm_type=comm_type,
            comm_alg=comm_alg,
        )


        (
            expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
        ) = torch_npu.npu_moe_distribute_dispatch_teardown(
            x=x,
            y=y,
            expert_ids=expert_ids,
            comm_cmd_info=comm_cmd_info,
            group_ep=group_ep,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            expert_shard_type=expert_shard_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            quant_mode=quant_mode,
            global_bs=global_bs,
            expert_token_nums_type=1,
            comm_type=comm_type,
            comm_alg=comm_alg,
        )

        if expand_x.dtype == torch.int8:
            expand_x = expand_x.to(input_dtype)
        if is_continue_golden_expand_x:
            expand_x = golden_expand_x

        quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(
            expand_x=expand_x,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            group_ep=group_ep,
            ep_world_size = ep_world_size,
            ep_rank_id = ep_rank_id,
            moe_expert_num=moe_expert_num,
            expert_shard_type=0,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            global_bs=global_bs,
            comm_quant_mode=comm_quant_mode,
            comm_type=comm_type,
            comm_alg=comm_alg,

        )


        output = torch_npu.npu_moe_distribute_combine_teardown(
            expand_x=expand_x,
            quant_expand_x=quant_expand_x,
            expert_ids=expert_ids,
            expand_idx=expand_idx,
            expert_scales=expert_scales,
            comm_cmd_info=comm_cmd_info,
            group_ep=group_ep,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            x_active_mask=x_active_mask,
            shared_expert_x=None,
            expert_shard_type=0,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            global_bs=global_bs,
            comm_quant_mode=comm_quant_mode,
            comm_type=comm_type,
            comm_alg=comm_alg,
        )
        return output


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
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks_list[0])
        if rank in ep_ranks:
            ep_group = tmp_group


    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    return ep_hcomm_info, ep_group


def get_dispatch_kwargs(
        x, expert_ids, x_active_mask, group_ep,
        ep_world_size, ep_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, 
        moe_expert_num, scales, quant_mode, global_bs, comm_type, expert_token_nums_type, comm_alg
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask else None
    if is_quant and is_quant_scales:
        scales = scales.to(torch.float32).npu()

    return {
        'x': x,
        'expert_ids': expert_ids,
        'x_active_mask': x_active_mask,
        'group_ep': group_ep,
        'ep_world_size': ep_world_size,
        'ep_rank_id': ep_rank_id,
        'expert_shard_type': expert_shard_type,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'moe_expert_num': moe_expert_num,
        'scales': scales,
        'quant_mode': quant_mode,
        'global_bs': global_bs,
        'comm_type': comm_type,
        'expert_token_nums_type': expert_token_nums_type,
        'comm_alg': comm_alg,
    }

def get_dispatch_setup_kwargs(
    x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask,
    expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs,comm_type, comm_alg
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask else None
    if is_quant and is_quant_scales:
        scales = scales.to(torch.float32).npu()
    return {
        'x': x,
        'expert_ids': expert_ids,
        'group_ep': group_ep,
        'ep_world_size': ep_world_size,
        'ep_rank_id': ep_rank_id,
        'moe_expert_num': moe_expert_num,
        'scales': scales,
        'x_active_mask': x_active_mask,
        'expert_shard_type': expert_shard_type,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'quant_mode': quant_mode,
        'global_bs': global_bs,
        'comm_type': comm_type,
        'comm_alg': comm_alg,
        'y_dtype':get_dtype_int_value(y_dtype)
    }

def get_dispatch_teardown_kwargs(
        # x, y, expert_ids, comm_cmd_info, group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type,
        # shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_type, comm_alg
        # x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type,
        # shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_type, comm_alg
        x, y, expert_ids, comm_cmd_info, group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type,
        shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_type, comm_alg
    ):
    x = x.to(input_dtype).npu()
    y = y.npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    comm_cmd_info = comm_cmd_info.to(torch.int32).npu()
    return {
        'x': x,
        'y': y,
        'expert_ids' : expert_ids,
        'comm_cmd_info': comm_cmd_info,
        'group_ep': group_ep,
        'ep_world_size': ep_world_size,
        'ep_rank_id': ep_rank_id,
        'moe_expert_num': moe_expert_num,
        'expert_shard_type': expert_shard_type,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'quant_mode': quant_mode,
        'global_bs': global_bs,
        'expert_token_nums_type': expert_token_nums_type,
        'comm_type': comm_type,
        'comm_alg': comm_alg,
    }


def get_combine_kwargs(
        expand_x, expert_ids, expand_idx, assist_info_for_combine, x_active_mask, expert_scales,
        group_ep, ep_world_size, ep_rank_id, expert_shard_type, 
        shared_expert_rank_num, moe_expert_num,comm_quant_mode, global_bs, shared_expert_num,
        shared_expert_x, comm_type, comm_alg
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask else None
    expert_scales = expert_scales.to(torch.float32).npu()
    assist_info_for_combine = assist_info_for_combine.to(torch.int32).npu()

    return {
        'expand_x': expand_x,
        'expert_ids': expert_ids,
        'expand_idx': expand_idx,
        'assist_info_for_combine': assist_info_for_combine,
        'x_active_mask': x_active_mask,
        'expert_scales': expert_scales,
        'group_ep': group_ep,
        'ep_world_size': ep_world_size,
        'ep_rank_id': ep_rank_id,
        'expert_shard_type': expert_shard_type,
        'shared_expert_x': None,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'moe_expert_num': moe_expert_num,
        'comm_quant_mode': comm_quant_mode,
        'global_bs': global_bs,
        'comm_type': comm_type,
        'comm_alg': comm_alg,
    }

def get_combine_setup_kwargs(
    expand_x, expert_ids, assist_info_for_combine,shared_expert_rank_num,global_bs, comm_quant_mode,
    group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, comm_alg, comm_type
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    assist_info_for_combine = assist_info_for_combine.to(torch.int32).npu()

    return {
        'expand_x': expand_x,
        'expert_ids': expert_ids,
        'assist_info_for_combine': assist_info_for_combine,
        'group_ep': group_ep,
        'ep_world_size': ep_world_size,
        'ep_rank_id': ep_rank_id,
        'moe_expert_num': moe_expert_num,
        'expert_shard_type': expert_shard_type,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'global_bs': global_bs,
        'comm_quant_mode': comm_quant_mode,
        'comm_type': comm_type,
        'comm_alg': comm_alg,
    }

def get_combine_teardown_kwargs(
    expand_x, quant_expand_x, expert_ids, expand_idx, expert_scales,ep_world_size,shared_expert_x, expert_shard_type,
    x_active_mask, comm_cmd_info, group_ep, ep_rank_id, moe_expert_num, shared_expert_num, shared_expert_rank_num,
    global_bs,comm_quant_mode, comm_type, comm_alg
):
    expand_x = expand_x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu()
    expert_scales = expert_scales.to(torch.float32).npu()
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask else None
    quant_expand_x = quant_expand_x.to(torch.int8).npu()
    comm_cmd_info = comm_cmd_info.to(torch.int32).npu()


    return {
        'expand_x': expand_x,
        'quant_expand_x': quant_expand_x,
        'expert_ids': expert_ids,
        'expand_idx': expand_idx,
        'expert_scales': expert_scales,
        'comm_cmd_info': comm_cmd_info,
        'group_ep': group_ep,
        'ep_world_size': ep_world_size,
        'ep_rank_id': ep_rank_id,
        'moe_expert_num': moe_expert_num,
        'x_active_mask': x_active_mask,
        'shared_expert_x': None,
        'expert_shard_type': 0,
        'shared_expert_num': shared_expert_num,
        'shared_expert_rank_num': shared_expert_rank_num,
        'global_bs': global_bs,
        'comm_quant_mode': comm_quant_mode,
        'comm_type': comm_type,
        'comm_alg': comm_alg,
    }

def run_dispatch_npu(queue, rank, x, expert_ids, x_active_mask, scales, shared_expert_num): #tjn
# def run_dispatch_npu(rank, x, expert_ids, x_active_mask, scales, shared_expert_num):
    set_device(rank)
    ep_hcomm_info, ep_group = init_hccl_comm(rank)

    print(f'[INFO] device_{rank} 构造dispatch算子输入数据')
    # print("start0120 x=",x.size(),
    #     "expert_ids=",expert_ids,
    #     "x_active_mask=",x_active_mask,
    #     "group_ep=",ep_hcomm_info,
    #     "moe_expert_num=",moe_expert_num,
    #     "scales=",scales,
    #     "quant_mode=",quant_mode,
    #     "global_bs=",global_bs,)
    dispatch_kwargs = get_dispatch_kwargs(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=x_active_mask,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        scales=scales,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        expert_token_nums_type=1,
        comm_alg=comm_alg,
    )
    dispatch_setup_kwargs = get_dispatch_setup_kwargs(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=x_active_mask,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        scales=scales,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        comm_alg=comm_alg,
    )
    dispatch_teardown_kwargs = get_dispatch_teardown_kwargs(
        x=x,
        y=x,
        comm_cmd_info=x,
        expert_ids=expert_ids,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        expert_token_nums_type=1,
        comm_alg=comm_alg,
    )

    if graph_type == 0:
        if not is_performance:
            for i in range(loop_acc):
                # print("zxye run_dispatch_npu acc i=", i, ", rank=", rank)
                dispatch_teardown_kwargs['x'] = x.npu()
                (
                    y, expand_idx, comm_cmd_info
                ) = torch_npu.npu_moe_distribute_dispatch_setup(**dispatch_setup_kwargs)
                dispatch_teardown_kwargs['y'] = y
                dispatch_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                # print("zxye after npu_moe_distribute_dispatch_setup acc i=", i, ", rank=", rank)
                (
                    expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
                ) = torch_npu.npu_moe_distribute_dispatch_teardown(**dispatch_teardown_kwargs)
                
                del dispatch_teardown_kwargs['x']
                gc.collect()
                torch.npu.empty_cache()
                #print("expandx size is ",expand_x.size(),"rankId is ", rank)
        else:
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(loop, dispatch_profiling_path) as prof:
                for i in range(loop):
                    # print("zxye performance run_dispatch_npu i=", i, ", rank=", rank)
                    torch.npu.synchronize()
                    (
                        y, expand_idx, comm_cmd_info
                    ) = torch_npu.npu_moe_distribute_dispatch_setup(**dispatch_setup_kwargs)

                    # print("zxye performance npu_moe_distribute_dispatch_setup i=", i, ", rank=", rank)
                    #dist.all_reduce(dispatch_teardown_kwargs['x'][0,0], op=ReduceOp.SUM, async_op=False, group=ep_group)
                    dispatch_teardown_kwargs['y'] = y
                    dispatch_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                    
                    (
                        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
                    ) = torch_npu.npu_moe_distribute_dispatch_teardown(**dispatch_teardown_kwargs)
                    print(f"Current memory allocated: {torch_npu.npu.memory_allocated()/1024**2:.2f} MB")
                    print(f"Max memory allocated: {torch_npu.npu.max_memory_allocated()/1024**2:.2f} MB")
                    # if i == 1:
                    #     for _ in range(100):
                    #         torch.exp(tensor)
                    prof.step()

    else:
        
        model = compile_model(MOE_DISTRIBUTE_DISPATCH_GRAPH_Model())

        if not is_performance:
            expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, expand_idx = model(**dispatch_kwargs)
        else:
            prof = get_profiling(loop, dispatch_profiling_path)
            prof.start()

            for _ in range(loop):
                torch.npu.synchronize()
                expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, expand_idx = model(**dispatch_kwargs)
                prof.step()

            prof.stop()
    torch.npu.synchronize()
    print(f'rank {rank} epid {rank} npu finished! \n')


    # 获得expand_x, 和dynamic_scales的真实值，并放入队列
    
    real_rank_token_num = expert_token_nums.cpu().sum().item()
    # print("expert_token_nums", expert_token_nums)
    y = y[:real_rank_token_num * (shared_expert_num + moe_expert_num), :]
    expand_x = expand_x[:real_rank_token_num, :] # tjn：截断
    # print("截断expandxsize",expand_x.size())
    # expand_x = expand_x[:real_rank_token_num, :] # tjn：截断
    # print("real_ranktokennum is",real_rank_token_num,"rank is",rank)
    # print("expandxsizeis", expand_x.size())
    dynamic_scales = dynamic_scales[:real_rank_token_num]
    # print(f"y_shape = {y.shape}\n")
    queue.put([
        rank,
        [
            y.cpu(), expand_x.cpu(), dynamic_scales.cpu(), expand_idx.cpu(),
            expert_token_nums.cpu(), assist_info_for_combine.cpu()
        ]
        # [
        #     y.cpu(),  expand_idx.cpu()
        # ]
    ]) #tjn


def run_combine_npu(
        queue, rank, expand_x, expert_ids, expand_idx, assist_info_for_combine, x_active_mask, expert_scales, shared_expert_num
):
    set_device(rank)
    ep_hcomm_info, ep_group = init_hccl_comm(rank)

    print(f'[INFO] device_{rank} 构造combine算子输入数据')
    print("run combine npu")
    combine_kwargs = get_combine_kwargs(
        expand_x=expand_x,
        expert_ids=expert_ids,
        expand_idx=expand_idx,
        assist_info_for_combine=assist_info_for_combine,
        x_active_mask=x_active_mask,
        expert_scales=expert_scales,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_x=None,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        comm_quant_mode=comm_quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        comm_alg= comm_alg,
    )

    combine_setup_kwargs = get_combine_setup_kwargs(
        expand_x=expand_x,
        expert_ids=expert_ids,
        assist_info_for_combine=assist_info_for_combine,
        group_ep=ep_hcomm_info,
        ep_world_size = ep_world_size,
        ep_rank_id = rank,
        moe_expert_num=moe_expert_num,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        comm_quant_mode=comm_quant_mode,
        comm_type=comm_type,
        comm_alg=comm_alg,
    )

    combine_teardown_kwargs = get_combine_teardown_kwargs(
        expand_x=expand_x,
        quant_expand_x=expand_x,
        expert_ids=expert_ids,
        expand_idx=expand_idx,
        expert_scales=expert_scales,
        comm_cmd_info=expand_x,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        x_active_mask=x_active_mask,
        shared_expert_x=None,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        comm_quant_mode=comm_quant_mode,
        comm_type=comm_type,
        comm_alg=comm_alg,
        
    )


    if graph_type == 0:
        if not is_performance:
            for _ in range(loop_acc):
                quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(**combine_setup_kwargs)
                combine_teardown_kwargs['quant_expand_x'] = quant_expand_x
                combine_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                x = torch_npu.npu_moe_distribute_combine_teardown(**combine_teardown_kwargs)
        else:
            quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(**combine_setup_kwargs)
            combine_teardown_kwargs['quant_expand_x'] = quant_expand_x
            combine_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
            x = torch_npu.npu_moe_distribute_combine_teardown(**combine_teardown_kwargs)

            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(loop, combine_profiling_path) as prof:
                for i in range(loop):
                    torch.npu.synchronize()
                    # 做全卡同步，因为combine只会同步一部分卡
                    quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(**combine_setup_kwargs)
                    dist.all_reduce(combine_kwargs['expand_x'][0, 0], op=ReduceOp.SUM, async_op=False, group=ep_group)
                    combine_teardown_kwargs['quant_expand_x'] = quant_expand_x
                    combine_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                    torch_npu.npu_moe_distribute_combine_teardown(**combine_teardown_kwargs)
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)

                    prof.step()

    else:

        model = compile_model(MOE_DISTRIBUTE_COMBINE_GRAPH_Model())
        if not is_performance:
            x = model(**combine_kwargs)
        else:
            prof = get_profiling(loop, combine_profiling_path)
            prof.start()

            for _ in range(loop):
                torch.npu.synchronize()
                x = model(**combine_kwargs)
                prof.step()

            prof.stop()
    torch.npu.synchronize()
    print(f'rank {rank} epid {rank} npu finished! \n')

    # 获得x中的有效token
    if is_mask:
        real_tokens = x_active_mask.cpu().sum().item()
        x = x[:real_tokens, :]
    queue.put([
        rank,
        [
            x.cpu()
        ]
    ])


def run_both_npu(queue, rank, x, expert_ids, x_active_mask, scales, expert_scales, golden_expand_x, shared_expert_num):
    set_device(rank)
    ep_hcomm_info, _ = init_hccl_comm(rank)

    print(f'[INFO] device_{rank} 构造两算子输入数据')
    expert_scales = expert_scales.to(torch.float32).npu()  # 图模式下传参需要
    golden_expand_x = golden_expand_x.to(input_dtype).npu()

    dispatch_kwargs = get_dispatch_kwargs(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=x_active_mask,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        scales=scales,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        expert_token_nums_type=1,
        comm_alg=comm_alg,
    )

    dispatch_setup_kwargs = get_dispatch_setup_kwargs(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=x_active_mask,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        scales=scales,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        comm_alg=comm_alg,
    )
    dispatch_teardown_kwargs = get_dispatch_teardown_kwargs(
        x=x,
        y=x,
        comm_cmd_info=x,
        expert_ids=expert_ids,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        quant_mode=quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        expert_token_nums_type=1,
        comm_alg=comm_alg,
    )

    combine_kwargs = get_combine_kwargs(
        expand_x=x,
        expert_ids=expert_ids,
        expand_idx=x,
        assist_info_for_combine=x,
        x_active_mask=x_active_mask,
        expert_scales=expert_scales,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        expert_shard_type=0,
        shared_expert_x=None,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        moe_expert_num=moe_expert_num,
        comm_quant_mode=comm_quant_mode,
        global_bs=global_bs,
        comm_type=comm_type,
        comm_alg= comm_alg,
    )

    combine_setup_kwargs = get_combine_setup_kwargs(
        expand_x=x,
        expert_ids=expert_ids,
        assist_info_for_combine=x,
        group_ep=ep_hcomm_info,
        ep_world_size = ep_world_size,
        ep_rank_id = rank,
        moe_expert_num=moe_expert_num,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        comm_quant_mode=comm_quant_mode,
        comm_type=comm_type,
        comm_alg=comm_alg,
    )

    combine_teardown_kwargs = get_combine_teardown_kwargs(
        expand_x=x,
        quant_expand_x=x,
        expert_ids=expert_ids,
        expand_idx=x,
        expert_scales=expert_scales,
        comm_cmd_info=x,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moe_expert_num,
        x_active_mask=x_active_mask,
        shared_expert_x=None,
        expert_shard_type=0,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        comm_quant_mode=comm_quant_mode,
        comm_type=comm_type,
        comm_alg=comm_alg,
        
    )

    if graph_type == 0:
        if not is_performance:
            for _ in range(loop_acc):
                (
                    y, expand_idx, comm_cmd_info
                ) = torch_npu.npu_moe_distribute_dispatch_setup(**dispatch_setup_kwargs)
                dispatch_teardown_kwargs['y'] = y
                dispatch_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                (
                    expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
                ) = torch_npu.npu_moe_distribute_dispatch_teardown(**dispatch_teardown_kwargs)
                # if is_quant:
                #     expand_x = expand_x.to(input_dtype)
                # torch.save(golden_expand_x[6], "golden_continue6.pt")
                if is_continue_golden_expand_x:
                    expand_x = golden_expand_x
                else:
                    if is_quant:
                        expand_x = convert_expand_x_dtype(expand_x)
                    # torch.save(expert_token_nums, "expert_token_nums.pt")
                    #real_rank_token_num = expert_token_nums.sum().item() \
                    real_rank_token_num = expert_token_nums.cpu().long().sum().item() \
                            if expert_token_nums_type == 1 \
                            else expert_token_nums[-1].item()
                    expand_x[real_rank_token_num:, :] = 0
                # torch.save(expand_x[6], "continue6.pt")				
                combine_setup_kwargs['expand_x'] = expand_x
                combine_setup_kwargs['assist_info_for_combine'] = assist_info_for_combine
                quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(**combine_setup_kwargs)
                combine_teardown_kwargs['expand_x'] = expand_x
                combine_teardown_kwargs['quant_expand_x'] = quant_expand_x
                combine_teardown_kwargs['expand_idx'] = expand_idx
                combine_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                x = torch_npu.npu_moe_distribute_combine_teardown(**combine_teardown_kwargs)
        else:
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(loop, continue_profiling_path) as prof:
                for i in range(loop):
                    (
                        y, expand_idx, comm_cmd_info
                    ) = torch_npu.npu_moe_distribute_dispatch_setup(**dispatch_setup_kwargs)
                    dispatch_teardown_kwargs['y'] = y
                    dispatch_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                    (
                        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
                    ) = torch_npu.npu_moe_distribute_dispatch_teardown(**dispatch_teardown_kwargs)
                    # if is_quant:
                    #     expand_x = expand_x.to(input_dtype)
                    if is_continue_golden_expand_x:
                        expand_x = golden_expand_x
                    else:
                        if is_quant:
                            expand_x = convert_expand_x_dtype(expand_x)
                        real_rank_token_num = expert_token_nums.sum().item() \
                                if expert_token_nums_type == 1 \
                                else expert_token_nums[-1].item()
                        expand_x[real_rank_token_num:, :] = 0
                    combine_setup_kwargs['expand_x'] = expand_x
                    combine_setup_kwargs['assist_info_for_combine'] = assist_info_for_combine
                    quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(**combine_setup_kwargs)
                    combine_teardown_kwargs['expand_x'] = expand_x
                    combine_teardown_kwargs['quant_expand_x'] = quant_expand_x
                    combine_teardown_kwargs['expand_idx'] = expand_idx
                    combine_teardown_kwargs['comm_cmd_info'] = comm_cmd_info
                    x = torch_npu.npu_moe_distribute_combine_teardown(**combine_teardown_kwargs)
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)

                    prof.step()

    else:
        continue_graph_kwargs = {
            **combine_kwargs,
            "x": x,
            "scales": scales,
            "quant_mode": quant_mode,
            "expert_token_nums_type": 1,
            "golden_expand_x": golden_expand_x,
        }

        model = compile_model(MOE_DISTRIBUTE_CONTINUE_GRAPH_Model())
        if not is_performance:
            x = model(**continue_graph_kwargs)
        else:
            
            prof = get_profiling(loop, continue_profiling_path)
            prof.start()

            for _ in range(loop):
                torch.npu.synchronize()
                x = model(**continue_graph_kwargs)
                prof.step()

            prof.stop()
    torch.npu.synchronize()
    print(f'rank {rank} epid {rank} npu finished! \n')
    
    
    # 获得x中的有效token
    if is_mask:
        real_tokens = x_active_mask.cpu().sum().item()
        x = x[:real_tokens, :]
    
    queue.put([
        rank,
        [
            x.cpu()
        ]
    ])

# 多进程
def gen_npu(target_func, **server_kwargs):
    def parse_rank_input(target_func, result_queue, rank, server_kwargs):
        def pad_expand_x(expand_x: torch.Tensor, ep_id: int):
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_x = torch.empty(size=[A - expand_x.size(0), h], dtype=expand_x.dtype)
            return torch.cat([expand_x, pad_for_expand_x])
            return torch.cat([expand_x])

        def filter_tensor(tensor: torch.Tensor, mask: torch.Tensor):
            return tensor[mask] if is_variable_bs else tensor

        ep_id = rank

        mask = server_kwargs["x_active_mask_list"][0][ep_id]

        if target_func == run_dispatch_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][0][ep_id], mask),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][0][ep_id], mask),
                "x_active_mask": server_kwargs["x_active_mask_list"][0][ep_id],
                "scales": server_kwargs["scales_list"][0],
                "shared_expert_num": shared_expert_num,
            }

        if target_func == run_combine_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "expand_x": pad_expand_x(server_kwargs["expand_x_list"][0][ep_id], ep_id),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][0][ep_id], mask),
                "expand_idx": server_kwargs["expand_idx_list"][rank],
                "assist_info_for_combine": server_kwargs["assist_info_for_combine"][rank],
                "x_active_mask": server_kwargs["x_active_mask_list"][0][ep_id],
                "expert_scales": filter_tensor(server_kwargs["expert_scales_list"][0][ep_id], mask),
                "shared_expert_num": shared_expert_num,
            }

        if target_func == run_both_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][0][ep_id], mask),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][0][ep_id], mask),
                "x_active_mask": server_kwargs["x_active_mask_list"][0][ep_id],
                "scales": server_kwargs["scales_list"][0],
                "expert_scales": filter_tensor(server_kwargs["expert_scales_list"][0][ep_id], mask),
                "golden_expand_x": pad_expand_x(server_kwargs["golden_expand_x_list"][0][ep_id], ep_id),
                "shared_expert_num": shared_expert_num,
            }

    if target_func not in [run_dispatch_npu, run_combine_npu, run_both_npu]:
        print("target_func not supported!")
        exit(0)

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

    # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表(当前存的是拼在一起的张量，后面会改)
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
        category_outputs.append(specific_category_output)
    # print("cate size is ",len(category_outputs))
    # print("cate0 size is ",category_outputs[1][0].size())
    # print("cate1 size is ",category_outputs[1][1].size())
    return category_outputs
    


# endregion
def gen_local_moe_expert_num():
    min_local_moe_expert_num = 1 if is_variable_local_moe_expert_num else local_moe_expert_num
    return [
        torch.randint(min_local_moe_expert_num, local_moe_expert_num + 1, size=[moe_rank_num])
        for _ in range(tp_world_size)
    ]

# region 输入构造函数
def gen_x(x_shape, dtype):
    x_list = []
    if is_x_random:
        cur_x = torch.empty(x_shape, dtype=dtype).uniform_(-5, 5)
    else:
        tmp = []
        for i in range(0, x_shape[0]):
            tmp.extend([i] * x_shape[1])
        cur_x = torch.tensor(tmp, dtype=dtype).view(x_shape)

    x_list.append(cur_x)
    return x_list


def gen_expert_ids():
    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]

    def gen_balanced_expert_ids_element(round_index):
        return torch.arange(round_index, round_index + moe_expert_num, dtype=torch.int32) % moe_expert_num

    ep_expert_ids_list = []
    if is_topk_balance:
        ep_expert_ids = [gen_balanced_expert_ids_element(round_index) for round_index in
                            range((bs * ep_world_size * k) // moe_expert_num + 1)]
    else:
        ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num) for _ in range(bs * ep_world_size)]
    ep_expert_ids_list.append(torch.cat(ep_expert_ids)[:bs * ep_world_size * k].view(expert_ids_shape))

    return ep_expert_ids_list


def gen_x_active_mask():
    def gen_card_active_mask():
        min_real_token_num = 0 if is_mask else 1
        real_token_num = torch.randint(min_real_token_num, bs + 1, (1,)).item() if is_unfixed_real_token else bs
        return torch.cat(
            (torch.ones(real_token_num), torch.zeros(bs - real_token_num))
        ).to(torch.bool)

    x_active_mask_list = []
    x_active_mask = [gen_card_active_mask() for _ in range(ep_world_size)]
    if is_variable_bs:
        upper_bs_card_ep_id = torch.randint(0, ep_world_size, (1,)).item()
        x_active_mask[upper_bs_card_ep_id] = torch.full((bs,), 1).to(torch.bool)
    x_active_mask_list.append(torch.cat(x_active_mask))

    return x_active_mask_list


def gen_scales():
    scales_list = []
    padded_scales_list = []

    if is_quant and is_quant_scales:
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
            scales = torch.empty(size=[shared_expert_num + moe_expert_num, h], dtype=torch.float32).uniform_(-1, 1)
            padded_scales = torch.cat((
                scales[:shared_expert_num, :].repeat_interleave(repeats=rank_num_per_shared_expert, dim=0),
                    scales[shared_expert_num:, :]
            ))
    else:
        # 使用空张量torch.tensor([])会被dispatch算子的tiling检查，只能用None，无法保证数据类型一致
        scales = None
        padded_scales = None
    scales_list.append(scales)
    padded_scales_list.append(padded_scales)

    return scales_list, padded_scales_list

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
    return [torch.ones(size=expert_ids_shape, dtype=torch.float32)]
    return [torch.empty(size=expert_ids_shape, dtype=torch.float32).uniform_(-1, 1)]

def gen_shared_expert_x(x_active_mask_list: List[List[torch.Tensor]], from_=0, to=1):
    if not is_shared_expert_x:
        return [torch.empty(0) for _ in range(ep_world_size)]
    
    x_active_mask_per_card = [chunk_tensor(mask_ep, ep_world_size) for mask_ep in x_active_mask_list]
    
    shared_expert_x_list = ([
        torch.empty(size=[x_active_mask_per_card[0][ep_id].sum().item(), h], dtype=input_dtype).uniform_(from_, to)
        for ep_id in range(ep_world_size)
    ])
    
    return shared_expert_x_list

def reshape_with_padding(input, tile_size):
    row, col = input.shape
    padded_col = ceil(col / tile_size) * tile_size
    input_padded = np.pad(input, 
                          ((0, 0), (0, padded_col - col)), 
                          mode='constant', 
                          constant_values=0)
    return input_padded.reshape(-1, tile_size)
# endregion

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

# region golden值生成函数
def gen_dispatch_golden(x_list, expert_ids_list, x_active_mask_list, padded_scales_list):

    def quant_process(ep_local_moe_expert_num: torch.Tensor, hared_list: list, moe_x_segments: list, scales):
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
            target_type = QUANT_MODE_DTYPE[y_dtype]
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
                #print("zxye before  y ",y.shape)
                dynamic_scales, y = mx_quantize(y, target_type, axis=-1, block_size=tile_size, round_mode="rint")
                #print("zxye dynamic_scales ",dynamic_scales)
                #print("zxye y ",y.shape)

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
        
        for rank_id in range(len(moe_x_segments)):
            moe_x_segments[rank_id], moe_dynamic_scales[rank_id] = cal_quant(
                moe_x_segments[rank_id],
                scales[shared_expert_rank_num+rank_id,:] if is_quant_scales else torch.ones((1,))
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



    def moe_dispatch(ep_local_moe_expert_num: torch.Tensor,
        x: torch.Tensor, expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
        ''' 
        将每张卡上的原始数据分发给moe专家。

        根据分发信息expert_ids，分发输入的张量x得到输出expanded_x，
        并生成expanded_x每行对应的（x, k）中元素的索引sorted_col_idx，
        以及分到各moe专家的数据数量golden_moe_token
        '''

        # 全局的x_mask掩掉x和expert_ids对应false的行，得到实际要发送的token的集合x及相应的expert_ids
        x = x[x_active_mask]
        expert_ids = expert_ids[x_active_mask]
        #GGprint("zxye x=", x)
        #print("zxye x_active_mask=", x_active_mask)
        #print("zxye expert_ids=", expert_ids)
        # expanded_x从前往后，放入x相应的行
        sorted_col_idx = torch.argsort(expert_ids.flatten(), stable=True)
        dst_row_to_src_row = sorted_col_idx // expert_ids.size(1)
        expanded_x = torch.index_select(input=x, dim=0, index=dst_row_to_src_row)

        # 统计每个专家出现次数，即为接收到的token数
        golden_moe_token = torch.bincount(expert_ids.flatten(), minlength=ep_local_moe_expert_num.sum().item())
        return expanded_x.to(input_dtype), sorted_col_idx, golden_moe_token


    def shared_dispatch(x: torch.Tensor, x_active_mask: torch.Tensor):
        '''
        将每张卡上的原始数据分发给共享专家。

        将 ep 域内所有卡分组 (ep group)，每组 rank_num_per_shared_expert 张卡，
        将共享专家卡按卡上专家分组 (shared group)，每组 rank_num_per_shared_expert 张卡，
        ep group 各组组内编号为 i 的卡将数据分发给 shared group 各组组内第 i 张共享专家卡
        '''

        shared_list = [torch.empty(0) for _ in range(shared_expert_rank_num)]
        shared_tokens = torch.empty((shared_expert_rank_num,), dtype=torch.int64)

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


    def cal_quant(y: torch.Tensor, scales):
            '''量化各条数据，并计算每条数据对应的scale

            根据各专家接收到的数据y和scales计算得到量化后的数据和量化场景的dynamic_scales
            '''
            # dynamic_scales一维，长度等于y的行数
            y = y.to(torch.float32) * scales

            y_max, _ = y.abs().max(dim=-1)
            dynamic_scales = y_max / 127.0
            dynamic_scales[dynamic_scales == 0] = 1.0  # scales = 0的时候改成1，避免除0
            y = (y / dynamic_scales.view(-1, 1)).round().to(torch.int8)

            return y, dynamic_scales
    
    def gen_y_out(x: torch.Tensor, x_active_mask: torch.Tensor, expert_ids: torch.Tensor):
        # 将x和expert_ids各自平均拆分为2份（对应两卡，chunks=2）
        card_x_list = chunk_tensor(x, ep_world_size)  # 拆分为2个张量
        card_expert_ids_list = chunk_tensor(expert_ids, ep_world_size)  # 拆分为2个张量
        # y_out_origin = x[x_active_mask].repeat_interleave(k+shared_expert_num, dim=0)
        all_card_results = []
        scalars = []
        # if is_quant:
        #     for i in range(y_out_origin.shape[0]):
        #         y_out_i, y_dynamic_scale = cal_quant(y_out_origin[i], scales[(expert_ids[i//(k+shared_expert_num)][i%(k+shared_expert_num)-1] + \
        #                 shared_expert_num)  if i%(k+shared_expert_num)!=0 else 0, :] if is_quant_scales else 1)
        #         y_out_i = y_out_i.view(-1)
        #         y_out_origin[i] = y_out_i
        #         scalars.append(y_dynamic_scale.unsqueeze(0))
        #     rows, orig_cols = y_out_origin.shape
        #     aligned32 = torch.nn.functional.pad(y_out_origin, pad=(0, (32-(orig_cols % 32)) % 32), mode='constant', value=0)
        #     concatenated = torch.cat([aligned32, torch.stack([s.view(torch.int8) for s in scalars])], dim=1)
        #     aligned_y_out = torch.nn.functional.pad(concatenated, pad=(0, (512-(concatenated.size(1) % 512)) % 512), mode='constant', value=0)

        # else:
        for card_idx in range(ep_world_size):
            expert_to_token_indices = {}
            # 取出当前卡的x和expert_ids（单卡数据，独立处理，不跨卡干扰）
            current_card_x = card_x_list[card_idx]  # 形状[4, 1024]
            current_card_expert_ids = card_expert_ids_list[card_idx]  # 形状[4, 8]
            num_tokens_per_card = current_card_x.shape[0]  # 单卡token个数：4
            for token_idx in range(num_tokens_per_card):
                # 获取当前token对应的所有专家ID（去重可选，根据需求保留原始重复）
                current_expert_ids = current_card_expert_ids[token_idx]
                
                for expert_id in current_expert_ids:
                    expert_id = expert_id.item()  # 转为Python原生整数
                    # 建立映射：专家ID对应token索引
                    if expert_id not in expert_to_token_indices:
                        expert_to_token_indices[expert_id] = []
                    expert_to_token_indices[expert_id].append(token_idx)
            # 步骤3：收集所有专家ID并进行全局升序排序
            all_expert_ids = sorted(expert_to_token_indices.keys())
            # 步骤4：展开排序后的专家ID，生成完整的专家ID序列（对应最终y的长度）
            sorted_expert_sequence = []
            for expert_id in all_expert_ids:
                # 每个专家ID对应的token个数，决定该专家ID在序列中出现的次数
                token_indices_for_expert = expert_to_token_indices[expert_id]
                sorted_expert_sequence.extend([expert_id] * len(token_indices_for_expert))
            # 步骤5：根据排序后的专家序列，映射填充对应的token，生成y张量
            y_list = []
            for expert_id in all_expert_ids:
                # 获取当前专家对应的所有token索引
                token_indices = expert_to_token_indices[expert_id]
                for token_idx in token_indices:
                    # 取出对应token并添加到y列表中
                    y_list.append(current_card_x[token_idx])
            
            # 步骤6：将列表转换为PyTorch张量（形状[64, 1024]）
            y_out_origin = torch.stack(y_list, dim=0)
            # 做字节对齐
            element_size = y_out_origin.element_size()
            origin_rows, origin_cols = y_out_origin.shape
            aligned_y_out = torch.zeros((origin_rows, ((origin_cols * element_size+511)//512)*512 // element_size), dtype=y_out_origin.dtype)
            aligned_y_out[:, :origin_cols] = y_out_origin
            all_card_results.append(aligned_y_out)

        # y_out = chunk_tensor(aligned_y_out, ep_world_size)
        # print(expert_ids)
        # print(sorted_expert_sequence)
        # print(y_out)
        return all_card_results
    # tjn量化
    # def quant_process(shared_list: list, moe_x_segments: list, scales):
    #     '''对各专家接收到的数据进行量化，并生成量化场景输出的scales
    #     '''
    #     # 每个专家的dynamic_scales，一个token一个scales
    #     shared_dynamic_scales = [torch.tensor([]) for _ in range(shared_expert_rank_num)]
    #     moe_dynamic_scales = [torch.tensor([]) for _ in range(moe_expert_num)]

    #     for i in range(shared_expert_rank_num):
    #         # dynamic_scales长度为shared_list[i] 行数，每个共享专家的token数，bs*(card_rate + 1)
    #         shared_list[i], shared_dynamic_scales[i] = cal_quant(shared_list[i], scales[i, :] if is_quant_scales else 1)

    #     ### 不用改，moe卡部分本来就是按照专家来写的
    #     for i in range(moe_expert_num):
    #         # dynamic_scales长度为golden_moe_token[i]大小，每个moe专家接收到的token数
    #         if moe_x_segments[i].size(0) > 0:
    #             moe_x_segments[i], moe_dynamic_scales[i] = cal_quant(
    #                 moe_x_segments[i],
    #                 scales[shared_expert_rank_num + i, :] if is_quant_scales else 1
    #             )

    #     # 得到各卡的dynamic scales
    #     moe_dynamic_scales_per_card = [
    #         torch.cat(moe_dynamic_scales[moe_expert_id: moe_expert_id + local_moe_expert_num])
    #         for moe_expert_id in range(0, moe_expert_num, local_moe_expert_num)]
    #     golden_dynamic_scales = shared_dynamic_scales + moe_dynamic_scales_per_card

    #     return shared_list, moe_x_segments, golden_dynamic_scales

    def gen_expand_idx(expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
        unfiltered_expert_idx_segments = chunk_tensor(expert_ids, ep_world_size)
        x_active_mask_per_card = chunk_tensor(x_active_mask, ep_world_size)
        expert_idx_segments = [segment[mask] for segment, mask in
                               zip(unfiltered_expert_idx_segments, x_active_mask_per_card)]

        global_occurrence_list = []
        for segment in expert_idx_segments:
            local_occurrence_list = []
            # 创建一个字典来统计当前卡的expert_ids中遇到每个数字之前它的出现次数
            occurrence_count_dict = {}
            # 遍历当前组内的每个元素
            for num in segment.flatten():
                num = num.item()
                if num in occurrence_count_dict:
                    occurrence_count_dict[num] += 1  # 如果已经存在，直接加 1
                else:
                    occurrence_count_dict[num] = 0  # 如果是第一次出现，初始化为 0
                # 将记录的次数存储到 count_tensor 中对应的位置
                local_occurrence_list.append(occurrence_count_dict[num])
            global_occurrence_list.append(torch.tensor(local_occurrence_list).to(torch.int32))

        return global_occurrence_list

    def gen_expand_idx_for_combine(expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
        """ 输出expand_x中各token对应的(rank_id, token_id, k_id)
        """

        def gen_shared_expand_idx(x_active_mask: torch.Tensor):
            shared_expand_idx = [[] for _ in range(shared_expert_rank_num)]       #存储每个共享专家收到的(r,t,k)

            real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

            for shared_card_id in range(shared_expert_rank_num):
                shared_expert_index = shared_card_id // rank_num_per_shared_expert
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                for send_card_id in range(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert):
                    rtk_list = [
                        ele
                        for token_id in range(real_token_num_list[send_card_id])
                        for ele in (send_card_id, token_id, k + shared_expert_index)
                    ]
                    shared_expand_idx[shared_card_id].extend(rtk_list)
            
            return shared_expand_idx

        def gen_moe_expand_idx(expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
            moe_expand_idx = [[] for _ in range(moe_expert_num)]  # 存储每个moe专家收到的(r,t,k)

            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)
            expert_ids_segments = expert_ids.chunk(ep_world_size)
            # 生成moe卡的expand_idx
            for send_rank_id in range(ep_world_size):
                segment = expert_ids_segments[send_rank_id]
                for token_id in range(bs):
                    if x_active_mask_per_card[send_rank_id][token_id]:
                        for k_id in range(k):
                            received_moe_expert = segment[token_id, k_id].item()
                            moe_expand_idx[received_moe_expert].extend([send_rank_id, token_id, k_id])

            # 转换，让moe_expand_idx存储每张moe卡收到的(r,t,k)
            moe_expand_idx = [sum(moe_expand_idx[moe_expert_id: moe_expert_id + local_moe_expert_num], [])
                              for moe_expert_id in range(0, moe_expert_num, local_moe_expert_num)]

            return moe_expand_idx

        expand_idx = gen_shared_expand_idx(x_active_mask) + gen_moe_expand_idx(expert_ids, x_active_mask)
        expand_idx = [torch.tensor(local_expand_idx) for local_expand_idx in expand_idx]  # 格式转换，以便后续处理
        return expand_idx

    def gen_assist_info(expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
        """生成各专家的assist_info
        """

        def gen_shared_expert_recv_count(x_active_mask: torch.Tensor):
            shared_expert_recv_count = torch.zeros(shared_expert_rank_num, ep_world_size, dtype=torch.int32)
            real_token_num_tensor = x_active_mask.view(-1, bs).sum(dim=1)

            for shared_card_id in range(shared_expert_rank_num):
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)
                shared_expert_recv_count[shared_card_id][send_card_ids] = real_token_num_tensor[send_card_ids]

            # 对每张共享专家卡上的recv_count求累加和，确定发回给每张卡的数据偏移和数据量
            shared_expert_recv_count = shared_expert_recv_count.cumsum(dim = 1)

            return shared_expert_recv_count

        def gen_moe_expert_recv_count(
            ep_local_moe_expert_num: torch.Tensor, 
            expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            # 将全局的expert_ids切分为ep_world_size份，放入元组expert_ids_segments中。每份为一张卡上的expert_ids
            original_expert_ids_segments = expert_ids.chunk(ep_world_size)
            expert_ids_segments = [
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
        shared_expert_recv_count = gen_shared_expert_recv_count(x_active_mask)
        moe_expert_recv_count = gen_moe_expert_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask)
        recv_count = [recv_count for recv_count in shared_expert_recv_count] + [recv_count for recv_count in
                                                                                moe_expert_recv_count]
        assist_info_for_combine = []
        for i, recv_count_tensor in enumerate(recv_count):
            assist_info_for_combine_half = recv_count_tensor
            padding_length = 128 * (A_shared if i < shared_expert_num * rank_num_per_shared_expert else A_moe) - assist_info_for_combine_half.size(0)
            zeros = torch.zeros(padding_length, dtype=assist_info_for_combine_half.dtype)
            assist_info_for_combine_half = torch.cat([assist_info_for_combine_half, zeros])
            assist_info_for_combine.append(assist_info_for_combine_half)
        return assist_info_for_combine
    

    golden_expand_x_list, golden_expand_idx_list, golden_tokens_list, \
        golden_assist_info_for_combine_list, golden_dynamic_scales_list, middle_idx_list, golden_expand_idx_for_combine_list = \
        [], [], [], [], [], [], []
    x, expert_ids, scales, x_active_mask = (
        x_list[0], expert_ids_list[0],
        padded_scales_list[0], x_active_mask_list[0]
    )

    for tp_id in range(tp_world_size):
        ep_local_moe_expert_num, x, expert_ids, scales, x_active_mask,  = (
            local_moe_expert_num_list[tp_id],
            x_list[tp_id], expert_ids_list[tp_id], 
            padded_scales_list[tp_id], x_active_mask_list[tp_id]
        )

    y_out_list = []
    y_out = gen_y_out(x, x_active_mask, expert_ids)
    y_out_list.append(y_out)
    #print("zxye y_out[0].shape=", y_out[0].shape) 

    # region ep域dispatch
    expanded_x, sorted_col_idx, golden_moe_token = moe_dispatch(ep_local_moe_expert_num, x, expert_ids, x_active_mask)
    #print("zxye golden_moe_token.tolist()=", golden_moe_token.tolist())
    #print("zxye expanded_x=", expanded_x)

    shared_list, shared_tokens = shared_dispatch(x, x_active_mask) if shared_expert_rank_num else ([], torch.empty(0))

    golden_tokens = torch.cat((shared_tokens, golden_moe_token))  # 分到所有卡上的golden值

    moe_x_segments = expanded_x.split(golden_moe_token.tolist())
    moe_x_segments = list(moe_x_segments)
    #print("zxye moe_x_segments[0].shape=", moe_x_segments[0].shape)
    golden_dynamic_scales = []
    if is_quant:
        shared_list, moe_x_segments, golden_dynamic_scales = quant_process(ep_local_moe_expert_num, shared_list, moe_x_segments, scales)
    moe_list = [torch.cat(moe_x_segments[moe_expert_id: moe_expert_id + local_moe_expert_num])
                for moe_expert_id in range(0, moe_expert_num, local_moe_expert_num)]
    golden_expand_x = shared_list + moe_list
    #print("moe_x_seg size is ",len(moe_x_segments))
    # print("golden_expand_x size0 is ",len(golden_expand_x[0]))
    # print("golden_expand_x size1 is ",len(golden_expand_x[1]))
    golden_expand_idx = gen_expand_idx(expert_ids, x_active_mask)

    golden_expand_idx_for_combine = gen_expand_idx_for_combine(expert_ids, x_active_mask)
    golden_assist_info_for_combine = gen_assist_info(expert_ids, x_active_mask)
    # endregion

    golden_expand_x_list.append(golden_expand_x)
    golden_expand_idx_list.append(golden_expand_idx)
    golden_tokens_list.append(golden_tokens)
    golden_assist_info_for_combine_list.append(golden_assist_info_for_combine)
    golden_dynamic_scales_list.append(golden_dynamic_scales)
    middle_idx_list.append(sorted_col_idx)
    golden_expand_idx_for_combine_list.append(golden_expand_idx_for_combine)

    return golden_expand_x_list, golden_expand_idx_list, golden_tokens_list, golden_assist_info_for_combine_list, golden_dynamic_scales_list, middle_idx_list, golden_expand_idx_for_combine_list, y_out_list



def gen_stack_golden(golden_expand_idx_list):
    golden_expand_idx = []
    for ep_id in range(ep_world_size):
        golden_expand_idx.append(golden_expand_idx_list[0][ep_id])
    return golden_expand_idx

def gen_splited_expandx_golden(expand_x):
    middle_shared_x = expand_x[:shared_expert_rank_num]
    middle_moe_x = expand_x[shared_expert_rank_num:]

    return (
        [torch.cat(middle_shared_x) if is_shared else torch.tensor([])],
        [torch.cat(middle_moe_x)],
    )


def gen_combine_golden(
        middle_shared_x_list, middle_moe_x_list, middle_idx_list,
        expert_scales_list, golden_tokens_list, x_active_mask_list, golden_expand_idx_list
):
    def comm_quant_round_trip(tensor: torch.Tensor):
        """ 将张量x做comm_quant量化，并对量化的结果进行逆运算

            comm_quant_mode=2时，对输入x进行comm_quant操作并返回结果
            comm_quant_mode=0时，直接返回输入x
        """

        def comm_quant(tensor: torch.Tensor):
            block_size = 32  # 核函数侧，一个 block 有 32 字节
            computation_dtype = torch.float32

            computation_dtype_size = torch.finfo(computation_dtype).bits // 8  # 用于量化计算的数据类型的字节数
            element_per_block = block_size // computation_dtype_size
            # 8

            quant_dtype = torch.int8 if comm_quant_mode == 2 else torch.int16
            quant_dtype_max = 127.0 if comm_quant_mode == 2 else 2047.0
            # 127,0

            blocks = tensor.view(-1, element_per_block).to(torch.float32)
            blocks_max, _ = blocks.abs().max(dim=-1, keepdim=True)
            blocks_scales = blocks_max / quant_dtype_max
            blocks_scales[blocks_scales == 0.0] = 1.0  # block_scales = 0 的时候改成 1，避免除 0
            blocks_quant = (blocks / blocks_scales).to(torch.float16).round().to(quant_dtype)

            return blocks_quant, blocks_scales.to(input_dtype)

        def comm_dequant(blocks_quant: torch.Tensor, blocks_scales: torch.Tensor):
            blocks_scales = blocks_scales.to(torch.float32)
            blocks_quant = blocks_quant.to(torch.float16).to(torch.float32)
            blocks_dequant = (blocks_quant * blocks_scales).to(input_dtype)
            tensor_dequant = blocks_dequant.view(-1, h)

            return tensor_dequant

        return comm_dequant(*comm_quant(tensor)) if comm_quant_mode != 0 else tensor

    def combine_x_for_shared(
            middle_shared_x: torch.Tensor, golden_tokens: torch.Tensor,
            x_active_mask: torch.Tensor, expand_idx: list,
            shared_scales=1
    ):
        '''根据expand_idx，将共享专家卡上的数据返回原来的卡
        '''

        real_expand_idx_per_card = [
            expand_idx[ep_id].split(3)
            for ep_id in range(ep_world_size)
        ]
        x_active_mask_per_card = x_active_mask.chunk(ep_world_size)
        middle_shared_x_per_card = middle_shared_x.split(golden_tokens[:shared_expert_rank_num].tolist())

        ori_real_x = [
            torch.empty(size=[local_active_mask.sum().item(), h], dtype=torch.float32)
            for local_active_mask in x_active_mask_per_card
        ]

        for shared_card_id in range(shared_expert_rank_num):
            local_middle_shared_x = middle_shared_x_per_card[shared_card_id]
            local_token_num = golden_tokens[shared_card_id].item()
            local_expand_idx = real_expand_idx_per_card[shared_card_id]
            for local_token_id in range(local_token_num):
                source_rank_id = local_expand_idx[local_token_id][0]
                source_token_id = local_expand_idx[local_token_id][1]
                ori_real_x[source_rank_id][source_token_id] = local_middle_shared_x[local_token_id]

        returned_tensor_after_round_trip = comm_quant_round_trip(torch.cat(ori_real_x))
        return returned_tensor_after_round_trip.to(torch.float32) * shared_scales

    def combine_x_for_moe(
            middle_moe_x: torch.Tensor,
            middle_idx: torch.Tensor,
            expert_scales: torch.Tensor,
            x_active_mask: torch.Tensor
    ):
        '''将真实的expand_x还回去，得到真实的x
        '''

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
        # 每个原始token对应k个处理后的token，将这些处理后的token按照k_id从前往后相加，确保顺序和核函数求和顺序一致
        scaled_x_k_by_token = scaled_x_k.view(-1, k, scaled_x_k.size(1))
        scaled_x = torch.zeros([scaled_x_k_by_token.size(0), scaled_x_k_by_token.size(-1)], dtype=torch.float32)
        for k_id in range(k):
            scaled_x += scaled_x_k_by_token[:, k_id, :]

        return scaled_x

    golden_x_list = []
    middle_shared_x, middle_moe_x, middle_idx, expert_scales, golden_tokens, x_active_mask, expand_idx = (
        middle_shared_x_list[0],
        middle_moe_x_list[0],
        middle_idx_list[0],
        expert_scales_list[0],
        golden_tokens_list[0],
        x_active_mask_list[0],
        golden_expand_idx_list[0],
    )

    golden_x_from_shared = (
        combine_x_for_shared(middle_shared_x, golden_tokens, x_active_mask, expand_idx)
        if is_shared else torch.tensor([])
    ).to(torch.float32)
    golden_x_from_moe = combine_x_for_moe(
        middle_moe_x,
        middle_idx,
        expert_scales,
        x_active_mask
    ).to(torch.float32)

    golden_x_ep = golden_x_from_moe
    if is_shared:
        golden_x_ep += golden_x_from_shared
    golden_x_ep = golden_x_ep.to(input_dtype)

    real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()
    golden_x_list.append(split_tensor(golden_x_ep, real_token_num_list))

    # golden_x为列表，存储每张卡上真实的x值
    golden_x = []
    for ep_id in range(ep_world_size):
        golden_x.append(golden_x_list[0][ep_id])

    return golden_x


# endregion
# tjn
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
        # print("dtype: ", dtype)
        np_output_npu = convert_to_numpy(npu, dtype)
        np_output_golden = convert_to_numpy(golden, dtype)

        max_diff_hd = 0.0
        if dtype in ("int8", "float32", "float8_e5m2", "float8_e4m3fn", "float8_e8m0", "hifloat8"):
            max_diff_hd = 1.0
        # res_output = toolss.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd,
        #                                     max_diff_hd=max_diff_hd) #tjn
        res_output = toolss.data_compare_np(np_output_npu, np_output_golden, diff_thd=diffThd, pct_thd=pctThd,
                                            max_diff_hd=max_diff_hd)
        print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
        write_row_to_csv('moe_distribute_dispatch.csv', [current_case_name, tensor_name, res_output])
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
        for i in range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev):
            print(f"##################Compare {mark_str} in rank", i, "epId", i, "##################")
            local_rank_id = i - server_index * rank_per_dev

            res = compare(
                golden_per_card[local_rank_id],
                npu_per_card[local_rank_id],
                compare_dtype,
                f"{compared_output.value}_{i}"
            )
            result[f"{compared_output.value}_{i}"] = res

    return result
# def compare_output(compared_output: ComparedOutput, golden_per_card, npu_per_card, compare_dtype, compare_per_rank):
#     def compare(golden, npu, dtype, tensor_name, diffThd=0.001, pctThd=0.001):
#         np_output_npu = np.array(golden.cpu())
#         np_output_golden = np.array(npu.cpu())


#         max_diff_hd = 0.0
#         if dtype == torch.int8:
#             max_diff_hd = 1.0
#         res_output = toolss.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd,
#                                             max_diff_hd=max_diff_hd)
#         print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
#         write_row_to_csv('moe_distribute_dispatch.csv', [current_case_name, tensor_name, res_output])
#         return res_output[0] == 'PASS'

#     mark_str = compared_output.value.upper().replace('_', ' ')
#     result = {}

#     if not compare_per_rank:
#         print(f"##################Compare {mark_str}##################")
#         res = compare(
#             torch.cat(golden_per_card).to(compare_dtype),
#             torch.cat(npu_per_card).to(compare_dtype),
#             compare_dtype,
#             compared_output.value
#         )
#         result[compared_output.value] = res
#     else:
#         for i in range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev):
#             print(f"##################Compare {mark_str} in rank", i, "epId", i, "##################")
#             local_rank_id = i - server_index * rank_per_dev

#             res = compare(
#                 golden_per_card[local_rank_id].to(compare_dtype),
#                 npu_per_card[local_rank_id].to(compare_dtype),
#                 compare_dtype,
#                 f"{compared_output.value}_{i}"
#             )
#             result[f"{compared_output.value}_{i}"] = res

#     return result



def generate_unique_random_numbers(m,k):
    """生成0-m-1范围内k个不重复、无固定顺序的随机整数"""
    # 定义数值范围：range(0, m) 表示 0、1、2...m-1（左闭右开特性）
    number_range = range(0, m)
    # 核心逻辑：sample从序列中随机选k个不重复元素，返回随机顺序的列表
    unique_random_nums = random.sample(number_range, k)
    return unique_random_nums

def sort_indices_by_values(values):
    """
    根据值的大小对序号升序排序，值相同保持原序号顺序
    :param values: 一维数值列表
    :return: 排序后的序号列表
    """
    index_value_pairs = list(enumerate(values))
    sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1])
    return [pair[0] for pair in sorted_pairs]
    
def count_prev_occurrences(arr):
    """
    计算一维数组中每个元素的值在其之前出现的次数
    :param arr: 原始一维数组
    :return: 每个位置对应的值的前置出现次数列表
    """
    # 字典：记录每个值已经出现的次数（动态更新）
    value_count = {}
    # 结果列表：存储每个位置的前置出现次数
    result = []
    
    # 遍历数组的每个元素（同时获取索引和值，索引可省略但保留更清晰）
    for idx, val in enumerate(arr):
        # 步骤1：获取当前值在之前出现的次数（不存在则为0）
        count = value_count.get(val, 0)
        result.append(count)
        
        # 步骤2：更新字典，当前值的计数+1（供后续元素统计使用）
        value_count[val] = count + 1
    #print(value_count)
    return result,value_count

def generate_y(x, m, use_num_dict):
    """
    根据给定数组x、整数m和次数字典use_num_dict生成满足所有规则的数组y
    新增规则：y中每个数字的出现次数严格匹配use_num_dict
    
    参数:
        x (list): 长度为n的数组，元素是0~n-1的不重复整数
        m (int): y数组元素的上界（y[i] < m）
        use_num_dict (dict): 键为y的元素值，值为对应出现次数
    
    返回:
        list: 满足所有规则的数组y
    """
    # 基础输入校验
    n = len(x)
    if m <= 0:
        raise ValueError("m必须是正整数")
    if not all(0 <= num < n for num in x) or len(set(x)) != n:
        raise ValueError("x不符合规则：元素需是0<=x[i]<n且不重复")
    
    # 校验use_num_dict的合法性
    # 1. 键必须是0<=key<m的整数
    for k in use_num_dict.keys():
        if not isinstance(k, int) or not (0 <= k < m):
            raise ValueError(f"use_num_dict的键{k}非法，需是0<=k<{m}的整数")
    # 2. 值必须是非负整数
    for v in use_num_dict.values():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"use_num_dict的值{v}非法，需是非负整数")
    # 3. 所有值的总和必须等于n
    total_count = sum(use_num_dict.values())
    if total_count != n:
        raise ValueError(f"use_num_dict的值总和{total_count}≠数组长度{n}")
    
    # 步骤1：生成基础temp数组（非递减，次数严格匹配use_num_dict）
    temp = []
    # 按数字升序添加，保证temp基础是非递减的
    for num in sorted(use_num_dict.keys()):
        temp.extend([num] * use_num_dict[num])
    
    # 步骤2：识别所有x[i]>x[i+1]的关键索引
    critical_indices = []
    for i in range(n-1):
        if x[i] > x[i+1]:
            critical_indices.append(i)
    
    # 步骤3：调整temp数组，确保关键位置满足temp[i] < temp[i+1]，且次数不变
    for i in critical_indices:
        if temp[i] >= temp[i+1]:
            # 需要让temp[i+1] > temp[i]，且保持总次数不变
            target_val = temp[i] + 1
            # 检查target_val是否在use_num_dict中（保证有足够的数量调整）
            if target_val not in use_num_dict or use_num_dict[target_val] == 0:
                raise ValueError(f"无法满足严格递增要求：关键位置{i}需要{target_val}，但use_num_dict中无该数字或次数为0")
            
            # 1. 将temp[i+1]改为target_val
            old_val = temp[i+1]
            temp[i+1] = target_val
            # 2. 在i+2之后找第一个target_val，改为old_val（补偿次数）
            # 保证总次数：old_val的次数不变，target_val的次数也不变
            compensate_pos = -1
            for j in range(i+2, n):
                if temp[j] == target_val:
                    compensate_pos = j
                    break
            if compensate_pos == -1:
                raise ValueError(f"无法补偿次数：关键位置{i}调整后，找不到{target_val}进行次数补偿")
            temp[compensate_pos] = old_val
    
    # 步骤4：根据temp和x生成y数组
    y = [0] * n
    for i in range(n):
        y[x[i]] = temp[i]
    
    # 验证y的次数是否匹配（可选）
    y_count = {}
    for num in y:
        y_count[num] = y_count.get(num, 0) + 1
    if y_count != use_num_dict:
        raise RuntimeError("y的数字次数与use_num_dict不匹配，调整逻辑异常")
    
    return y
    
def generate_non_negative_list(k: int, target: int) -> list:
    """
    生成长度为k、总和等于target且所有元素均为非负数（≥0）的列表
    
    参数:
        k: 列表长度（非负整数）
        target: 列表元素的目标总和（非负整数）
        
    返回:
        满足条件的非负整数列表
        
    异常:
        - k为负数时，抛出ValueError
        - target为负数时，抛出ValueError
        - k=0但target≠0时，抛出ValueError
    """
    # 基础合法性校验：k和target必须非负
    if not isinstance(k, int) or k < 0:
        raise ValueError("列表长度k必须是非负整数")
    if not isinstance(target, int) or target < 0:
        raise ValueError("目标总和target必须是非负整数")
    
    # 处理边界情况1：空列表
    if k == 0:
        if target == 0:
            return []
        else:
            raise ValueError("长度为0的列表总和只能是0")
    
    # 处理边界情况2：列表长度为1
    if k == 1:
        return [target]
    
    # 生成前k-1个非负整数，确保它们的和 ≤ target（避免最后一个数为负）
    first_part = []
    remaining = target  # 剩余可分配的数值，保证前k-1个数的和不超target
    for _ in range(k-1):
        # 每个数的上限是剩余可分配值（最后一个数至少留0）
        max_val = remaining
        num = random.randint(0, max_val)
        first_part.append(num)
        remaining -= num  # 剩余值减少，确保后续数的和不超target
    
    # 计算最后一个数，保证总和等于target且非负
    last_element = target - sum(first_part)
    result_list = first_part + [last_element]
    
    return result_list


def pad_tensors_end(golden_expand_x, npu_expand_x):
    """
    将较小n维度的tensor在末尾补零，对齐到较大n维度
    
    Args:
        golden_expand_x: list of tensors, 每个tensor形状为(n_golden, 1024)
        npu_expand_x: list of tensors, 每个tensor形状为(n_npu, 1024)
    
    Returns:
        golden_expand_x_padded, npu_expand_x_padded: 对齐后的两个列表
    """
    assert len(golden_expand_x) == len(npu_expand_x), "两个列表长度必须相同"
    print("2520npuexpand5 size is ",npu_expand_x[0].size()," ",npu_expand_x[1].size())
    golden_expand_x_padded = []
    npu_expand_x_padded = []
    
    for golden_tensor, npu_tensor in zip(golden_expand_x, npu_expand_x):
        n_golden = golden_tensor.shape[0]
        n_npu = npu_tensor.shape[0]
        max_n = max(n_golden, n_npu)
        
        # 对golden_tensor进行padding（在n维度的末尾补零）
        if n_golden < max_n:
            # 创建要补零的部分
            zeros_to_add = torch.zeros(max_n - n_golden, 1024,
                                      dtype=golden_tensor.dtype,
                                      device=golden_tensor.device)
            # 在末尾（dim=0）连接零值
            golden_padded = torch.cat([golden_tensor, zeros_to_add], dim=0)
        else:
            golden_padded = golden_tensor
        
        # 对npu_tensor进行padding（在n维度的末尾补零）
        if n_npu < max_n:
            zeros_to_add = torch.zeros(max_n - n_npu, 1024,
                                      dtype=npu_tensor.dtype,
                                      device=npu_tensor.device)
            npu_padded = torch.cat([npu_tensor, zeros_to_add], dim=0)
        else:
            npu_padded = npu_tensor
        
        golden_expand_x_padded.append(golden_padded)
        npu_expand_x_padded.append(npu_padded)
    
    return golden_expand_x_padded, npu_expand_x_padded

if __name__ == "__main__":
    # 遗留文件清理
    clean_previous_profiling_result()
    print_setting()
    check_setting()

    # 设置ep域的全局x和expert_ids的形状
    x_shape = [bs * ep_world_size, h]
    expert_ids_shape = [bs * ep_world_size, k]

    # 设置随机种子
    torch.manual_seed(random_seed)

    # 生成各ep域的x
    x_list = gen_x(x_shape, input_dtype)
    # 生成各ep域的expert_ids
    expert_ids_list = gen_expert_ids()
    # print("expert_ids_list[0] = ", expert_ids_list[0])
    # expert_ids_list=[torch.tensor([
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [12, 13, 14, 15, 16, 17],
    #     [18, 19, 20, 21, 22, 23],
    #     [24, 25, 26, 27, 28, 29],
    #     [30, 31,  1,  2,  3,  4],
    #     [ 5,  6,  7,  8,  9, 10],
    #     [11, 12, 13, 14, 15, 16],
    #     [17, 18, 19, 20, 21, 22],
    #     [23, 24, 25, 26, 27, 28],
    #     [29, 30, 31,  0,  2,  3],
    #     [ 4,  5,  6,  7,  8,  9],
    #     [10, 11, 12, 13, 14, 15],
    #     [16, 17, 18, 19, 20, 21],
    #     [22, 23, 24, 25, 26, 27],
    #     [28, 29, 30, 31,  0,  1],
    #     [ 3,  4,  5,  6,  7,  8],
    #     [ 9, 10, 11, 12, 13, 14],
    #     [15, 16, 17, 18, 19, 20],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31,  0],
    #     [ 1,  2,  4,  5,  6,  7],
    #     [ 8,  9, 10, 11, 12, 13],
    #     [14, 15, 16, 17, 18, 19],
    #     [20, 21, 22, 23, 24, 25],
    #     [26, 27, 28, 29, 30, 31],
    #     [ 0,  1,  2,  3,  5,  6],
    #     [ 7,  8,  9, 10, 11, 12],
    #     [13, 14, 15, 16, 17, 18],
    #     [19, 20, 21, 22, 23, 24],
    #     [25, 26, 27, 28, 29, 30],
    #     [31,  0,  1,  2,  3,  4]])]
    # expert_ids_list=[torch.tensor([
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [ 0,  1,  2,  3,  4,  5],
    #     [ 6,  7,  8,  9, 10, 11],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26],
    #     [21, 22, 23, 24, 25, 26],
    #     [27, 28, 29, 30, 31, 26]])]
    #expert_ids_list=[torch.tensor([[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],
    #[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],
    #[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21]
    #,[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21]])]
    # expert_ids_list=[torch.tensor([[4,5,6,7],[4,5,6,7],[4,5,6,7],[4,5,6,7],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]])]
    # expert_ids_list=[torch.tensor([[4],[5],[6],[7],[0],[1],[2],[3]])]
    # expert_ids_list=[torch.tensor([[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21]
    # ,[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],
    # [0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],
    # [0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]])]

    #print(x_list[0])
    

    # if ep_world_size==2:
    #     x_half=x_list[0][:bs]
    #     x_list=[x_half.repeat(2,1)]

    #     expert_ids_1=expert_ids_list[0][-bs:].flatten().tolist()
    #     first_sort_result_1=sort_indices_by_values(expert_ids_1)
    #     expandIdx_1,count_dict_1=count_prev_occurrences(expert_ids_1)

    #     count_dict_0=dict()
    #     rank_1_to_0_num=0
    #     for key,value in count_dict_1.items():
    #         if key<moe_expert_num//2:
    #             rank_1_to_0_num+=value
    #             count_dict_0[key+moe_expert_num//2]=value
    #         else:
    #             count_dict_0[key-moe_expert_num//2]=value

    #     expert_ids_0=generate_y(first_sort_result_1[rank_1_to_0_num:]+first_sort_result_1[:rank_1_to_0_num],moe_expert_num,count_dict_0)
    #     expert_ids=expert_ids_0+expert_ids_1
    #     expert_ids_list=[torch.tensor(expert_ids).reshape(-1,k)]

    #print(x_list[0][:bs])
    #print(x_list[0][-bs:])
    #print(expert_ids_list[0])
    # 生成各ep域的moe专家分布
    local_moe_expert_num_list = gen_local_moe_expert_num()
    # 生成各ep域的x_active_mask及各ep域有效token的个数
    x_active_mask_list = gen_x_active_mask()
    # 生成各ep域的scales
    #scales_list, padded_scales_list = gen_scales()
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
        golden_assist_info_for_combine_list, golden_dynamic_scales_list, middle_idx_list, golden_expand_idx_for_combine_list, y_out_list
    ) = gen_dispatch_golden(x_list, expert_ids_list, x_active_mask_list, padded_scales_list)
    y_out, golden_expand_x, golden_assist_info_for_combine, golden_tokens, golden_dynamic_scales, golden_expand_idx = (
        y_out_list[0],
        golden_expand_x_list[0],
        golden_assist_info_for_combine_list[0],
        golden_tokens_list[0],
        golden_dynamic_scales_list[0],
        golden_expand_idx_list[0],
    )
    # expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list]
    # print(expert_ids_list)
    # sys.exit(0)
    golden_tokens = divide_per_card(golden_tokens, world_size, 1)
    # endregion

    # 生成各ep域的expert_scales
    expert_scales_list = gen_expert_scales()

    # region 生成combine算子golden值

    # if is_quant:
    #     golden_expand_x_list[0] = [tensor.to(input_dtype) for tensor in golden_expand_x_list[0]]
    if is_quant:
        for tp_id in range(tp_world_size):
            golden_expand_x_list[tp_id] = [convert_expand_x_dtype(tensor) for tensor in golden_expand_x_list[tp_id]]
    # torch.save(golden_expand_x_list[0], "golden_expand_x_list.pt")
    middle_shared_x_list, middle_moe_x_list = gen_splited_expandx_golden(golden_expand_x_list[0])

    golden_x = gen_combine_golden(
        middle_shared_x_list, middle_moe_x_list,
        middle_idx_list, expert_scales_list,
        golden_tokens_list, x_active_mask_list,
        golden_expand_idx_for_combine_list
    )

    # endregion

    print("Generate golden success.\n")


    if is_dispatch:
        [npu_y_out, npu_expand_x, npu_dynamic_scales, npu_expand_idx, npu_tokens, npu_assist_info_for_combine] = gen_npu(
            run_dispatch_npu,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in x_active_mask_list],
            scales_list=scales_list,
            shared_expert_num=shared_expert_num,
        )
        # print("2520npuexpand0 size is ",npu_expand_x[0].size()," ",npu_expand_x[1].size())
        
        # npu_setup_dynamic_scales = []
        # npu_y_out_align_list = []
        # print("npu_y_out.dtype",npu_y_out[0].dtype)
        # print("npu_y_out.shape",npu_y_out[0].shape)
        # for i in range(len(npu_y_out)):
        #     # npu_y_cut = npu_y_out[i][:, :h].clone()
        #     npu_y_cut = copy.deepcopy(npu_y_out[i][:, :h])
        #     # print(f"npu_y_cut.dtype = {npu_y_cut.dtype}")
        #     print(npu_y_cut)
        #     if quant_mode == 2:
        #         npu_scales = npu_y_out[i][:, Align(h,32):(Align(h,32) + 4)]
        #     elif quant_mode == 3:
        #         npu_scales = npu_y_out[i][:, Align(h,128):(Align(h,128) + ceil(h / 128) * 4)]
        #     elif quant_mode == 4:
        #         npu_scales = npu_y_out[i][:, Align(h,256):Align((Align(h,256) + ceil(h / 32)),2)]
        #     print("=" * 60)
        #     print("步骤1：截取完成")
        #     print(f"截取后张量{i}形状：{npu_scales.shape}")
        #     print(f"截取后张量{i}类型：{npu_scales.dtype}")
        #     # 步骤2.1：先将 float8_e4m3fn 转为 uint8 视图（1 字节/元素，无损保留二进制）
        #     npu_scales_uint8 = npu_scales.view(torch.uint8)

        #     # 步骤2.2：将 uint8 视图转为 float32 类型，再 reshape 为 (2, 128)
        #     if quant_mode == 2 or quant_mode == 3:
        #         print(f"npu_scales_uint8.shape[1] = {npu_scales_uint8.shape[1]}")
        #         npu_scales_view = npu_scales_uint8.view(torch.float32).reshape(npu_scales_uint8.shape[0],int(npu_scales_uint8.shape[1] / 4))
        #     elif quant_mode == 4:
        #         npu_scales_view = npu_scales_uint8

        #     print("=" * 60)
        #     print("步骤2：view 转为 (2, 128) float32 完成")
        #     print(f"最终张量{i}形状：{npu_scales_view.shape}")
        #     print(f"最终张量{i}类型：{npu_scales_view.dtype}")

        #     if quant_mode != 0:
        #         npu_setup_dynamic_scales.append(npu_scales_view)
        #     torch.save(npu_y_cut, f"my_tensor{i}.pt")
        #     npu_y_out_align_list.append(npu_y_cut)



        if is_performance:
            dispatch_performance, dispatch_performace_memory_list = get_dispatch_performance()
            # print("print_dispatch_performance2696:",dispatch_performance)

        # 精度对比
        print("\n\n##################Start Dispatch Compare##################")

        result_dispatch = {}

        #print("npu_y_out = ", npu_y_out)
        #print("npu_expand_x = ", npu_expand_x)

        #print("zxye golden_expand_x[server_ranks]= ", golden_expand_x[server_ranks])
        # #比较 y_out
        # res = compare_output(
        #     ComparedOutput.Y_OUT, 
        #     y_out[server_ranks],
        #     npu_y_out,
        #     # torch.int8 if is_quant else input_dtype,
        #     y_dtype, # tjn
        #     True,
        # )
        #result_dispatch.update(res)
        # print("2520npuexpand1 size is ",npu_expand_x[0].size()," ",npu_expand_x[1].size())


        # 比较 assist_info_for_combine
        res = compare_output(
            ComparedOutput.EP_ASSIST_INFO,
            [t[:ep_world_size if i < shared_expert_num else local_moe_expert_num * ep_world_size] for i, t in enumerate(golden_assist_info_for_combine[server_ranks])],
            [t[:ep_world_size if i < shared_expert_num else local_moe_expert_num * ep_world_size] for i, t in enumerate(npu_assist_info_for_combine)],
            # torch.int32,
            "int32",
            True,
        )
        result_dispatch.update(res)

        # 比较 expand x
        # for i in range(golden_expand_x[0].size(0)):
        #     print("golden expandx0 is ",golden_expand_x[0][i][0],"\n")
        # for i in range(npu_expand_x[0].size(0)):
        #     print("npu expandx0 is ",npu_expand_x[0][i][0],"\n")
        # for i in range(golden_expand_x[1].size(0)):
        #     print("golden expandx1 is ",golden_expand_x[1][i][0],"\n")
        # for i in range(npu_expand_x[1].size(0)):
        #     print("npu expandx1 is ",npu_expand_x[1][i][0],"\n")
        # print("golden expandx0 is ",golden_expand_x[0][0])
        # print("npu expandx0 is ",npu_expand_x[0][0])
        # print("golden expandx1 is ",golden_expand_x[1][0])
        # print("npu expandx1 is ",npu_expand_x[1][0])
        # tempgolden, tempnpu = pad_tensors_end(golden_expand_x, npu_expand_x)
        print("2520npuexpand2 size is ",npu_expand_x[0].size()," ",npu_expand_x[1].size())
        print("2520goldenexpand2 size is ",golden_expand_x[0].size()," ",golden_expand_x[1].size())
        res = compare_output(
           ComparedOutput.EXPAND_X,
           golden_expand_x[server_ranks],
           npu_expand_x,
           #torch.int8 if is_quant else torch.float32,
           y_dtype,
           True,
        )
        result_dispatch.update(res)
        print("2520npuexpand3 size is ",npu_expand_x[0].size()," ",npu_expand_x[1].size())

        # 比较 dynamic scales
        if is_quant:
            if quant_mode == 4 and npu_dynamic_scales[0].dtype != torch.uint8:
                npu_dynamic_scales = [npu_dynamic_scales[i].view(torch.uint8) for i in range(len(npu_dynamic_scales))]
            res = compare_output(
                ComparedOutput.DYNAMIC_SCALES,
                golden_dynamic_scales[server_ranks],
                npu_dynamic_scales,
                # torch.float32,
                "float32",
                False
            )
            result_dispatch.update(res)

        #比较 expand idx
        res = compare_output(
            ComparedOutput.EXPAND_IDX,
            golden_expand_idx[server_ranks],
            npu_expand_idx,
            # torch.int32,
            "int32",
            False
        )
        result_dispatch.update(res)

        # 比较 expert token nums
        res = compare_output(
            ComparedOutput.EXPERT_TOKEN_NUMS,
            golden_tokens[server_ranks],
            npu_tokens,
            # torch.int64,
            "int64",
            False
        )
        result_dispatch.update(res)
    
    # torch.save(golden_expand_x_list[0], "com_bine_golden_expand_x_list.pt")
    if is_combine:
        [npu_x] = gen_npu(
            run_combine_npu,
            expand_x_list=golden_expand_x_list,
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            expand_idx_list=golden_expand_idx,
            assist_info_for_combine=golden_assist_info_for_combine,
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in x_active_mask_list],
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list],
            shared_expert_num=shared_expert_num,
        )

        print("Generate combine npu success.")

        if is_performance:
            combine_performance, combine_performance_memory_list = get_combine_performance()

        print("\n\n##################Start Combine Compare##################")

        result_combine = {}
        # 比较x
        res = compare_output(
            ComparedOutput.X,
            golden_x[server_ranks],
            npu_x,
            # torch.float32,
            "float32",
            True
        )
        result_combine.update(res)

    if is_continue:
        [npu_x] = gen_npu(
            run_both_npu,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in x_active_mask_list],
            scales_list=scales_list,
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list],
            golden_expand_x_list=golden_expand_x_list,
            shared_expert_num=shared_expert_num,
        )

        if is_performance:
            continue_performance, continue_performance_memory_list = get_continue_performance()

        print("Generate continue npu success.")

        print("\n\n##################Start Continue Compare##################")

        result_continue = {}
        # 比较 x
        res = compare_output(
            ComparedOutput.X,
            golden_x[server_ranks],
            npu_x,
            # torch.float32,
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

        if is_performance:
            # print("print_dispatch_performance2852:",dispatch_performance)
            print_dispatch_performance(dispatch_performance)
            
            index_i = 0
            for i in dispatch_performace_memory_list:
                if graph_type == 0:
                    print(path_to_Allocated_type[dispatch_profiling_path][index_i])
                print_dispatch_performance_memory(i)
                index_i += 1

    if is_combine:
        print("\n##################combine result##################")
        print("result: ", result_combine)
        all_result = all(result_combine.values())
        if all_result:
            print("combine test Success.")
        else:
            print("combine test Failed.")

        if is_performance:
            print_combine_performance(combine_performance)
            
            index_i = 0
            for i in combine_performance_memory_list:
                if graph_type == 0:
                    print(path_to_Allocated_type[combine_profiling_path][index_i])
                print_combine_performance_memory(i)
                index_i += 1

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
            
            index_i = 0
            for i in continue_performance_memory_list:
                if graph_type == 0:
                    print(path_to_Allocated_type[continue_profiling_path][index_i])
                print_continue_performance_memory(i)
                index_i += 1
    

