import os

script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "torch_extensions_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir

import csv
import time
import torch
import socket
import torch_npu
import traceback
import numpy as np
from torch.multiprocessing import Process, Manager, Lock, Value
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from math import ceil
from enum import Enum
import torchair
from torchair.configs.compiler_config import CompilerConfig
import argparse

import toolss_pro
from graph import CascadeGraphModel, MoeDispatchFFNCombineGraphModel
from gen_input_and_golden import GenInputAndGolden
from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, npu_get_mega_moe_ccl_buffer_size

def init_case_info(data):
    global graph_type, quant_mode, comm_quant_mode, is_quant, is_performance, loop, repeat, warmup
    global is_variable_bs, is_full_core, is_topk_balance, is_x_random, random_seed, is_mask, is_unfixed_real_token
    global x_active_mask_dim, server_num, tp_world_size, rank_per_dev, world_size, ep_world_size
    global server_index, master_ip, host_url_list, ep_ranks_list, tp_ranks_list, shared_expert_num
    global rank_num_per_shared_expert, shared_expert_rank_num, local_moe_expert_num, moe_rank_num
    global moe_expert_num, is_shared, core_num, aic_core_num, aiv_core_num, is_independent_group_buffersize
    global ep_buffersize, tp_buffersize, bs, h, n, k, global_bs, max_shared_group_num, A_shared, A_moe
    global out_dtype, current_case_name, quant_dtype_dict, swiglu_quant_dict, dispatch_quant_out_type_dict
    global swiglu_quant_dtype, quant_dtype, weight_dtype, dispatch_quant_out_type, scale_dtype
    global expert_scales_dtype, transpose_weight, is_graph, comm_alg, is_cloud, server_ranks
    global print_compare_details, buff_size, input_kwargs, cascade_graph_kwargs
    # region 控制参数配置

    # region 执行哪些测试

    x_active_mask_dim = 0  # mask的维度，当前支持0,1,2

    graph_type = int(data['graph_type'])  # 0为单算子，1为静态图模式，2为动态图模式, 3为aclgraph
    is_graph = (graph_type != 0)
    quant_mode = int(data['quant_mode'])  # 0为dispatch算子非量化场景，1为静态量化（当前不支持），2为动态量化
    comm_quant_mode = 0  # 0无量化，1则combine通信采用int12动态量化（不支持），2为int8动态量化, 3为e5m2, 4为e4m3
    is_independent_group_buffersize = 1
    ep_buffersize = 6000
    tp_buffersize = 1000

    is_performance = int(data['is_performance']) # 是否进行性能检测
    loop = 105  # 性能检测时的性能采集轮数
    repeat = 1  # 性能采集分为几个批次
    warmup = 0  # 每批次的热身轮数

    is_variable_bs = 0  # 是否使能可变bs
    is_full_core = True  # True时为全核模式，使用全部的核，False时使用部分核

    core_num = 18 if not is_full_core else 36  # 非全核模式使用的核数（单个核上有一个aic核，一个aiv核）

    is_quant = (quant_mode > 0)

    aic_core_num = core_num  # 非全核模式使用的aic核数
    aiv_core_num = core_num * 2  # 非全核模式使用的aiv核数
    server_num = 1  # 单机或者多机模式
    tp_world_size = 1  # 1或2, 2时dispatch算子在tp域执行allgather, combine算子执行reducescatter, 1时均不执行
    shared_expert_num = 0
    rank_num_per_shared_expert = 0  # 每个共享专家占据的卡数

    # 赋值给原来的变量名，后面的代码无需改动
    bs = int(data['bs'])
    local_moe_expert_num = int(data['expert_per_rank'])
    is_cloud = False  # 是否在云上环境进行测试，使用云上环境时设置为True，使用物理机时设为False
    if not is_cloud:
        server_index = 0 if server_num > 1 else 0  # 多机每个脚本保持一致并递增index值
        master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'  # 如果通信不通，查看一下master_ip是否设对
    else:
        # 云上环境用下面方式获取server_index和master_ip
        server_index = int(os.environ.get("VC_TASK_INDEX"))
        host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
        master_ip = socket.gethostbyname(host_url_list[0])

    rank_per_dev = int(data['ep_world_size'])  # 每个host有几个die
    is_mask = 1 if x_active_mask_dim == 1 else 0
    is_unfixed_real_token = is_mask or is_variable_bs  # 每张卡上真实token数是否固定
    world_size = server_num * rank_per_dev  # 通信域总die数
    ep_world_size = world_size // tp_world_size
    shared_expert_rank_num = shared_expert_num * rank_num_per_shared_expert  # 共享专家卡数
    moe_rank_num = ep_world_size - shared_expert_rank_num  # moe专家卡数
    moe_expert_num = moe_rank_num * local_moe_expert_num  # moe专家数

    is_shared = (shared_expert_rank_num > 0)


    h = int(data['h'])  # 每个token的长度
    n = int(data['n'])
    k = int(data['k'])
    comm_alg = ''

    transpose_weight = 1

    global_bs = bs * ep_world_size
    max_shared_group_num = ceil(ep_world_size / rank_num_per_shared_expert) if (rank_num_per_shared_expert > 0) else 0
    A_shared = bs * max_shared_group_num
    A_moe = global_bs * min(local_moe_expert_num, k)

    out_dtype = torch.bfloat16  # 控制dispatch输入及combine输出的数据类型， 0为bfloat16，1为float16

    quant_dtype_dict = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.uint8,  # float4存储为uint8 (2个float4打包成1个byte)
        # "float4_e1m2": torch.uint8,  # float4存储为uint8 (2个float4打包成1个byte)
    }

    swiglu_quant_dict = {
        "float8_e4m3fn": 24,
        "float8_e5m2": 23,
        "float4_e2m1": 296,
        # "float4_e1m2": 41,
    }

    dispatch_quant_out_type_dict = {
        "float8_e4m3fn": 24,
        "float8_e5m2": 23,
        "float4_e2m1": 296,
        # "float4_e1m2": 297,
    }
    swiglu_quant_dtype = str(data['swiglu_quant_dtype']).strip()  # 当前仅支持swiglu_quant_dict表量化模式，swiglu量化模式
    quant_dtype = str(data['quant_mode_type']).strip()    # 当前仅支持quant_dtype_dict表量化模式，dispatch量化模式
    weight_dtype = str(data['weight_dtype']).strip()   # 当前仅支持quant_dtype_dict表量化模式，weight量化模式
    dispatch_quant_out_type = dispatch_quant_out_type_dict[quant_dtype]
    scale_dtype = torch.float8_e8m0fnu # 当前仅支持torch.float8_e8m0fnu
    is_topk_balance = True
    is_x_random = True  # 是否让x值随机
    random_seed = int(data['seed'])  # 随机种子，用于固定各随机输入变量的产生，如x,expert_ids,expert_scales等
    expert_scales_dtype = torch.float32
    current_case_name = str(data['case_name']).strip()

    ep_ranks_list = [list(range(tp_id, world_size, tp_world_size)) for tp_id in range(tp_world_size)]
    tp_ranks_list = [list(range(ep_id * tp_world_size, (ep_id + 1) * tp_world_size)) for ep_id in range(ep_world_size)]
    server_ranks = slice(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)

    print_compare_details = False

    # buff_size = npu_get_mega_moe_ccl_buffer_size(ep_world_size, moe_expert_num, bs, k, h, quant_mode, dispatch_quant_out_type,
    #                                             comm_quant_mode, comm_alg)
    # os.environ['HCCL_BUFFSIZE'] = f'{buff_size}'

    print('======================================================')
    print(f'{current_case_name=}')
    # print(f'{buff_size=}')
    print(f'{bs=}')
    print(f'{h=}')
    print(f'{n=}')
    print(f'{k=}')
    print(f'{local_moe_expert_num=}')
    print(f'{random_seed=}')
    print(f'{dispatch_quant_out_type=}')
    print(f'{expert_scales_dtype=}')
    print(f'{quant_dtype=}')
    print(f'{out_dtype=}')
    print(f'{transpose_weight=}')
    print(f'{is_performance=}')
    print(f'{graph_type=}')
    print(f'{quant_mode=}')
    print(f'{swiglu_quant_dtype=}')
    print(f'{weight_dtype=}')
    print('======================================================')
    input_kwargs = {
        'bs': bs, 'h': h, 'n': n, 'quant_dtype': quant_dtype,
        'is_x_random': is_x_random, 'global_bs': global_bs, 'out_dtype': out_dtype, 'moe_expert_num': moe_expert_num,
        'is_topk_balance': is_topk_balance, 'k': k, 'is_mask': is_mask, 'is_unfixed_real_token': is_unfixed_real_token,
        'ep_world_size': ep_world_size, 'is_variable_bs': is_variable_bs, 'expert_scales_dtype': expert_scales_dtype,
        'local_moe_expert_num': local_moe_expert_num, 'quant_mode': quant_mode, 'transpose_weight': transpose_weight,
        'comm_quant_mode': comm_quant_mode, 'weight_dtype': weight_dtype, 'swiglu_quant_dtype': swiglu_quant_dtype
    }

    cascade_graph_kwargs = {
        'loop': loop, 'is_performance': is_performance, "aic_core_num": aic_core_num, "aiv_core_num": aiv_core_num,
        "is_graph": is_graph
    }

class ComparedOutput(Enum):
    EP_RECV_COUNTS = "ep_recv_counts"
    TP_RECV_COUNTS = "tp_recv_counts"
    EXPAND_X = "expand_x"
    EXPAND_IDX = "expand_idx"
    DYNAMIC_SCALES = "dynamic_scales"
    EXPERT_TOKEN_NUMS = "expert_token_nums"
    X = "x"
    Y1 = 'y1'
    Y1_SCALE = 'y1_scale'

# dispatch_profiling_path = './Profiling_Result_mc2_dispatch_16p'
# combine_profiling_path = './Profiling_Result_mc2_combine_16p'
cascade_profiling_path = './Profiling_Result_mc2_cascade_16p'
dispatch_op_type = 'MoeDistributeDispatchV3'
combine_op_type = 'MoeDistributeCombineV3'
matmul_swiglu_op_type = 'GroupedMatmulSwigluQuantV2'
mega_moe_op_type = 'MegaMoe'
path_to_op_types = {
    # dispatch_profiling_path: [dispatch_op_type],
    # combine_profiling_path: [combine_op_type],
    cascade_profiling_path: [dispatch_op_type, combine_op_type, matmul_swiglu_op_type, mega_moe_op_type]
}



def write_row_to_csv(csv_name: str, row_content: list):
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)

def check_input():
    if k > moe_expert_num:
        print('k大于总的moe专家数!')
        exit(0)

    if weight_dtype == "float4_e2m1" and quant_dtype != "float8_e4m3fn" and swiglu_quant_dtype != "float8_e4m3fn":
        print('当weight_dtype为float4_e2m1时，quant_dtype和swiglu_quant_dtype必须为float8_e4m3fn!')
        exit(0)
    
    if weight_dtype != "float4_e2m1" and (weight_dtype != quant_dtype):
        print('weight_dtype fp8类型必须与quant_dtype相同!')
        exit(0)

def get_torch_dtype(dtype):
    type_dict = {
        'bf16': torch.bfloat16, 'fp32': torch.float, 'fp16': torch.float16,
        'int8': torch.int8, 'int32': torch.int32, 'int64': torch.int64, 'uint64': torch.int64
    }
    type_dict.update({'float8_e4m3fn': torch.float8_e4m3fn, 'float8_e5m2': torch.float8_e5m2, 'float8_e8m0': torch_npu.float8_e8m0fnu,
    'hifloat8': torch_npu.hifloat8, 'float4_e2m1': torch_npu.float4_e2m1fn_x2,'float4_e1m2': torch_npu.float4_e1m2fn_x2})
    if isinstance(dtype, str):
        return type_dict.get(dtype.lower())
    else:
        return type_dict.get(dtype)

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

def clean_previous_profiling_result():
    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')

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

                    local_rank_id = int(
                        memory_record['Device Type'][0].split(":")[1])  # memory_record['Device Type']值形如 "NPU:6"
                    rank_id = rank_per_dev * server_index + local_rank_id

                    rank_max_allocated_memory = memory_record['Total Allocated(MB)'].max()
                    rank_min_allocated_memory = memory_record['Total Allocated(MB)'].min()
                    rank_memory_usage = rank_max_allocated_memory - rank_min_allocated_memory

                    memory_usage_per_rank.loc[memory_usage_per_rank.shape[0]] = [rank_id, rank_memory_usage]

        memory_usage_per_rank['rank_id'] = memory_usage_per_rank['rank_id'].astype('int32')
        memory_usage_per_rank.sort_values(by='rank_id', inplace=True)
        memory_usage_per_rank.reset_index(drop=True, inplace=True)
        return memory_usage_per_rank

    return get_execution_time(profiling_path), None
    # return get_execution_time(profiling_path), get_memory_usage(profiling_path)

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
        return max(get_dispatch_expected_memory_usage((ep_rank_id)),
                   get_combine_expected_memory_usage(ep_rank_id, tp_rank_id))

def print_cascade_performance(cascade_performance):
    (
        dispatch_share_performance, dispatch_moe_performance, combine_share_performance, combine_moe_performance,
        matmul_swiglu_share_performance, matmul_swiglu_moe_performance, dispatch_ffn_combine_share_performance,
        dispatch_ffn_combine_moe_performance,

        min_dispatch_share_performance, min_dispatch_moe_performance, min_combine_share_performance,
        min_combine_moe_performance, min_matmul_swiglu_share_performance, min_matmul_swiglu_moe_performance, 
        min_dispatch_ffn_combine_share_performance, min_dispatch_ffn_combine_moe_performance,
    ), memory_usage = cascade_performance

    print(
        f'[INFO] MC2融合算子Average性能为:'
        f'dispatch共享: {dispatch_share_performance} us, dispatch Moe: {dispatch_moe_performance} us, '
        f'combine共享: {combine_share_performance} us, combine Moe: {combine_moe_performance} us\n'
        f'matmul_swiglu共享:{matmul_swiglu_share_performance} us, matmul_swiglu Moe:{matmul_swiglu_moe_performance} us\n'
        f'dispatch_ffn_combine共享:{dispatch_ffn_combine_share_performance} us, dispatch_ffn_combine Moe:{dispatch_ffn_combine_moe_performance} us\n'
    )
    print(
        f'[INFO] MC2融合算子Min性能为:'
        f'dispatch共享:{min_dispatch_share_performance} us, dispatch Moe:{min_dispatch_moe_performance} us, '
        f'combine共享:{min_combine_share_performance} us, combine Moe:{min_combine_moe_performance} us\n'
        f'matmul_swiglu共享:{min_matmul_swiglu_share_performance} us, matmul_swiglu Moe:{min_matmul_swiglu_moe_performance} us\n'
        f'dispatch_ffn_combine共享:{min_dispatch_ffn_combine_share_performance} us, dispatch_ffn_combine Moe:{min_dispatch_ffn_combine_moe_performance} us\n'
    )
    cascade_all = dispatch_moe_performance+combine_moe_performance+matmul_swiglu_moe_performance
    dispatch_combine = dispatch_moe_performance+combine_moe_performance
    print(f'计算通信掩盖率提升为：{((cascade_all - dispatch_ffn_combine_moe_performance) / dispatch_combine):.2%}')
    print(f'[INFO] Dispatch与Combine算子总内存占用:')
    # print_dataframe(memory_usage)

def get_cascade_performance():
    cascade_performance, memory_usage = get_performance(cascade_profiling_path)

    print_cascade_performance([cascade_performance, memory_usage])

    dispatch_performance = cascade_performance[:2] + cascade_performance[8:10]
    combine_performance = cascade_performance[2:4] + cascade_performance[10:12]
    matmul_swiglu_performance = cascade_performance[4:6] + cascade_performance[12:14]
    dispatch_ffn_combine_performance = cascade_performance[6:8] + cascade_performance[14:16]
    op_types = path_to_op_types[cascade_profiling_path]
    save_performance_result('cascade', current_case_name, op_types, [dispatch_performance, combine_performance, matmul_swiglu_performance, dispatch_ffn_combine_performance],
                            memory_usage)

    return [cascade_performance, memory_usage]

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
    # save_memory_usage_result(test_mode, case_name, op_types, op_memory_usage_info)

def compile_model(model):
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(1)

    print(f"------------------graph_type: {graph_type}------------------", )
    compiler_config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)

    if graph_type == 3:
        print('is_aclgraph=1, 测试场景: aclgraph入图模式')
        compiler_config.mode = "reduce-overhead"     # compiler_config.mode在get_npu_backend之后可以生效
        compiled_model = torch.compile(model, backend=npu_backend)
    elif graph_type == 1:
        # 传统入图模式，静态shape+在线编译场景
        print('graph_type=1, 测试场景: 传统入图模式，静态shape+在线编译场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:
        # ACLNN入图模式，动态shape+二进制
        print('graph_type=2, 测试场景: ACLNN入图模式，动态shape+二进制场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)

    return compiled_model

def pack_float4_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x).view(np.uint8)
    x = x & 0x0F
    packed = x[..., 0::2] | (x[..., 1::2] << 4)
    return packed.astype(np.uint8)

def get_cascade_kwargs(rank, ep_hcomm_info, tp_hcomm_info, weight1_input, weight2_input, weight1_scale_input, weight2_scale_input,
                        x, expert_ids, expert_scales, x_active_mask):
    if weight_dtype == "float4_e2m1" or weight_dtype == "float4_e1m2":
        weight1 = weight1_input
        weight1 = torch.from_numpy(pack_float4_to_uint8(weight1))
        weight2 = weight2_input
        weight2 = torch.from_numpy(pack_float4_to_uint8(weight2))
    else:
        weight1 = weight1_input.tobytes()
        weight1 = torch.frombuffer(weight1, dtype=quant_dtype_dict[quant_dtype]).reshape(weight1_input.shape).clone()
        weight2 = weight2_input.tobytes()
        weight2 = torch.frombuffer(weight2, dtype=quant_dtype_dict[quant_dtype]).reshape(weight2_input.shape).clone()
    weight1_scale = weight1_scale_input.tobytes()
    weight1_scale = torch.frombuffer(weight1_scale, dtype=torch.float8_e8m0fnu).reshape(weight1_scale_input.shape).clone()
    weight2_scale = weight2_scale_input.tobytes()
    weight2_scale = torch.frombuffer(weight2_scale, dtype=torch.float8_e8m0fnu).reshape(weight2_scale_input.shape).clone()
    return [
        x.npu(), expert_ids.npu(), expert_scales.npu(),
        x_active_mask.npu(), ep_hcomm_info, tp_hcomm_info, rank, ep_world_size, moe_expert_num, quant_mode,
        global_bs, weight1.npu(), out_dtype, None if quant_mode==0 else weight1_scale.npu(), weight2.npu(),
        None if quant_mode==0 else weight2_scale.npu(), dispatch_quant_out_type, comm_quant_mode, rank, get_torch_dtype(weight_dtype) if weight_dtype.lower() in [
            'float4_e2m1','float4_e1m2','hifloat8'] else None, swiglu_quant_dict[swiglu_quant_dtype], get_torch_dtype(quant_dtype) if quant_dtype.lower() in [
            'float4_e2m1','float4_e1m2','hifloat8'] else None
    ]

def get_symm_buffer_kwargs(group_ep):
    return {
        "group": group_ep,
        "num_experts": moe_expert_num,
        "num_max_tokens_per_rank": global_bs,
        "num_topk": k,
        "hidden": h,
        "intermediate_hidden": h,
        "max_recv_token_num": bs * ep_world_size * min(local_moe_expert_num, k),
        "dispatch_quant_mode": quant_mode,
        "dispatch_quant_out_type": dispatch_quant_out_type,
        "combine_quant_mode": comm_quant_mode,
        "comm_alg": ""
        }

def get_dispatch_ffn_combine_kwargs(rank, distribute_buffer, weight1_input, weight2_input, weight1_scale_input, weight2_scale_input, 
                                    expert_ids, x, expert_scales, x_active_mask):
    if quant_dtype == "float4_e2m1" or quant_dtype == "float4_e1m2":
        weight1 = weight1_input
        weight1 = torch.from_numpy(pack_float4_to_uint8(weight1))
        weight2 = weight2_input
        weight2 = torch.from_numpy(pack_float4_to_uint8(weight2))
    else:
        weight1 = weight1_input.tobytes()
        weight1 = torch.frombuffer(weight1, dtype=quant_dtype_dict[quant_dtype]).reshape(weight1_input.shape).clone()
        weight2 = weight2_input.tobytes()
        weight2 = torch.frombuffer(weight2, dtype=quant_dtype_dict[quant_dtype]).reshape(weight2_input.shape).clone()
    weight1_scale = weight1_scale_input.tobytes()
    weight1_scale = torch.frombuffer(weight1_scale, dtype=torch.float8_e8m0fnu).reshape(weight1_scale_input.shape)
    weight2_scale = weight2_scale_input.tobytes()
    weight2_scale = torch.frombuffer(weight2_scale, dtype=torch.float8_e8m0fnu).reshape(weight2_scale_input.shape)
    return {
        "x": x.npu(),
        "topk_ids": expert_ids.npu(),
        "topk_weights": expert_scales.npu(),
        "l1_weights": weight1.npu(),
        "l2_weights": weight2.npu(),
        "sym_buffer": distribute_buffer,
        "l1_weights_sf": None if quant_mode==0 else weight1_scale.npu(),
        "l2_weights_sf": None if quant_mode==0 else weight2_scale.npu(),
        "x_active_mask": x_active_mask.npu(),
        "weight1_type": dispatch_quant_out_type_dict[weight_dtype],
        "weight2_type": dispatch_quant_out_type_dict[weight_dtype],
        "rank": rank,
    }

def gen_npu(target_func, excel, sheet, start_line, end_line):
    def parse_rank_input(target_func, rank, excel, sheet, start_line, end_line, shared_counters, lock):
        if target_func == run_cascade_npu:
            return {
                "rank": rank,
                "excel": excel,
                "sheet": sheet,
                "start_line": start_line,
                "end_line": end_line,
                "shared_counters": shared_counters,
                "lock":lock
            }

    if target_func != run_cascade_npu:
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
    mp.set_start_method("forkserver", force=True)
    shared_counters = [Value('i', 0) for _ in range(end_line - start_line + 1)]
    lock = Lock()

    for rank in rank_list:
        rank_kwargs = parse_rank_input(target_func, rank, excel, sheet, start_line, end_line, shared_counters, lock)
        proc = Process(target=target_func, kwargs=rank_kwargs)
        print(f"============================卡{rank} Start Proc============================")
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")

def init_hccl_comm_two_ep(rank):
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")
    
    print(f"device {rank} 初始化EP域")

    optionsep = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    optionsep.hccl_config = {"hccl_buffer_size":ep_buffersize}
    optionstp = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    optionstp.hccl_config = {"hccl_buffer_size":tp_buffersize}
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks, pg_options = optionsep)
        if rank in ep_ranks:
            ep_group_dispatch = tmp_group

    print(f"device {rank} 初始化EP-2域")
    for ep2_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep2_ranks)
        if rank in ep2_ranks:
            ep_group_mega = tmp_group

    print(f"device {rank} 初始化TP域")
    for tp_ranks in tp_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=tp_ranks, pg_options = optionstp)
        if rank in tp_ranks:
            tp_group_dispatch = tmp_group

    ep_hcomm_info_dispatch = ep_group_dispatch._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    ep2_hcomm_info_mega = ep_group_mega._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm_info_dispatch = tp_group_dispatch._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    
    return ep_hcomm_info_dispatch, tp_hcomm_info_dispatch, ep2_hcomm_info_mega, ep_group_dispatch, tp_group_dispatch, ep_group_mega

def run_cascade_npu(rank, excel, sheet, start_line, end_line, shared_counters, lock):
    data = read_case_info(excel, sheet, start_line - 1)
    init_case_info(data)
    set_device(rank)
    ep_hcomm_info_dispatch, tp_hcomm_info_dispatch, _, _, _, ep_group_mega = init_hccl_comm_two_ep(rank)
    distribute_buffer = get_symm_buffer_for_mega_moe(**get_symm_buffer_kwargs(ep_group_mega))
    tmp = 0
    for data_line in range(start_line - 1, end_line):
        try:
            data = read_case_info(excel, sheet, data_line)
            init_case_info(data)
            case_sub_dir = os.path.join("./input", sheet, current_case_name)
            try:
                id_part = [p for p in current_case_name.split('_') if p.startswith('ID')][0]
            except IndexError:
                id_part = current_case_name
            
            ep_id = rank // tp_world_size
            tp_id = rank % tp_world_size
            weight1 = torch.load(os.path.join(case_sub_dir, f"weight1.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            weight2 = torch.load(os.path.join(case_sub_dir, f"weight2.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            weight1_scale = torch.load(os.path.join(case_sub_dir, f"weight1_scale.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            weight2_scale = torch.load(os.path.join(case_sub_dir, f"weight2_scale.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            x = torch.load(os.path.join(case_sub_dir, f"x.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            expert_ids = torch.load(os.path.join(case_sub_dir, f"expert_ids.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            expert_scales = torch.load(os.path.join(case_sub_dir, f"expert_scales.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            x_active_mask = torch.load(os.path.join(case_sub_dir, f"x_active_mask.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            golden_expert_token_nums = torch.load(os.path.join(case_sub_dir, f"expert_token_nums.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]

            golden_x = torch.load(os.path.join(case_sub_dir, f"x_final.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
                    
            print(f'[INFO] device_{rank} 构造两算子输入数据')

            cascade_kwargs = get_cascade_kwargs(rank, ep_hcomm_info_dispatch, tp_hcomm_info_dispatch, weight1, weight2, weight1_scale, weight2_scale,
                        x, expert_ids, expert_scales, x_active_mask)
            
            distribute_buffer.num_experts = moe_expert_num,
            distribute_buffer.num_max_tokens_per_rank = global_bs,
            distribute_buffer.num_topk = k,
            distribute_buffer.hidden = h,
            distribute_buffer.intermediate_hidden = h,
            distribute_buffer.max_recv_token_num = bs * ep_world_size * min(local_moe_expert_num, k),
            distribute_buffer.dispatch_quant_mode = quant_mode,
            distribute_buffer.dispatch_quant_out_type = dispatch_quant_out_type,
            distribute_buffer.combine_quant_mode = comm_quant_mode,
            distribute_buffer.comm_alg = ""
            dispatch_ffn_combine_kwargs = get_dispatch_ffn_combine_kwargs(rank, distribute_buffer, weight1, weight2, weight1_scale, weight2_scale, 
                                    expert_ids, x, expert_scales, x_active_mask)
            print(f'rank:{rank} 构造算子输入finished!')
            if graph_type == 0:
                cascade_moe = CascadeGraphModel(**cascade_graph_kwargs)
                dispatch_ffn_combine_model = MoeDispatchFFNCombineGraphModel(**cascade_graph_kwargs)
                if not is_performance:
                    # cascade_graph_kwargs['distribute_buffer'] = distribute_buffer
                    x, dispatch_out = cascade_moe.forward(*cascade_kwargs)
                    y, expert_token_nums = dispatch_ffn_combine_model.forward(**dispatch_ffn_combine_kwargs)
                else:
                    
                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(100, 1, 0, cascade_profiling_path) as prof:
                        for i in range(100):
                            #torch.npu.synchronize()
                            x, dispatch_out = cascade_moe.forward(*cascade_kwargs)
                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)
                            prof.step()
                        torch.npu.synchronize()
                    with get_profiling(100, 1, 0, cascade_profiling_path) as prof:
                        for i in range(100):
                            #    torch.npu.synchronize()
                            y, expert_token_nums = dispatch_ffn_combine_model.forward(**dispatch_ffn_combine_kwargs)
                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)
                            prof.step()
            else:

                loop_time = 2 if is_performance and is_graph else 1
                if is_performance:
                    repeat_time = 1
                    warm_time = 1
                    prof = get_profiling(loop_time, repeat_time, warm_time, cascade_profiling_path)
                    prof.start()
            
                for i in range(loop_time):
                    # 调用小算子
                    cascade_moe = CascadeGraphModel(**cascade_graph_kwargs)
                    x, dispatch_out = cascade_moe.forward(*cascade_kwargs)

                    if is_performance:
                        prof.step()
                if is_performance:
                    prof.stop()
                
                dispatch_ffn_combine_model = compile_model(MoeDispatchFFNCombineGraphModel(**cascade_graph_kwargs))
                if is_performance:
                    repeat_time = 1
                    warm_time = 1
                    prof = get_profiling(loop_time, repeat_time, warm_time, cascade_profiling_path)
                    prof.start()
            
                for i in range(loop_time):
                    # 调用融合算子
                    y, expert_token_nums = dispatch_ffn_combine_model(**dispatch_ffn_combine_kwargs)
                    if is_performance:
                        prof.step()
                if is_performance:
                    prof.stop()

            torch.npu.synchronize()
            print(f'rank {rank} epid {rank//tp_world_size} tpid {rank%tp_world_size} npu finished! \n')
            # time.sleep(5 if not is_elastic else 40)
            expert_token_nums = expert_token_nums.cpu()
            x = x.cpu()
            y = y.cpu()

            if is_mask:
                real_tokens = input_data.x_active_mask.cpu().sum().item()
                x = x[:real_tokens, :]
                y = y[:real_tokens, :]
            time.sleep(1*rank)

            print("##################Start Continue Compare##################")
            result_dispatch_ffn_combine = {}
            # 比较y

            # 关闭级联算子精度对比：不再对 cascade_moe 输出 x 做三方对比
            # res = compare_output_triangle(
            #     ComparedOutput.X,
            #     x,
            #     golden_x,
            #     y,
            #     torch.float32,
            #     True,
            #     rank
            # )
            # result_combine.update(res)

            # 比较expert_token_num
            # golden_expert_token_nums = input_data.expert_token_nums

            res = compare_output_triangle(
                ComparedOutput.EXPERT_TOKEN_NUMS,
                'two comparison',
                golden_expert_token_nums,
                expert_token_nums,
                torch.float32,
                True,
                rank
            )
            result_dispatch_ffn_combine.update(res)
            print(f"##################Compare x in rank,{rank}##################")

            # 关闭级联算子精度对比：仅对融合算子 y 与 golden 做两方对比
            res_output = toolss_pro.data_compare_np(
                np.array(golden_x.cpu().to(torch.float32)),
                np.array(y.cpu().to(torch.float32)),
                diff_thd=0.001, pct_thd=0.001)
            res = 'PASS' if res_output[0] == 'PASS' else 'FAIL'
            print('[INFO] tensor y output 精度结果为：{}'.format(res))
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            write_row_to_csv('mega_moe_result.csv', [current_case_name, f"output_x_rank{rank}", res, time_str, args.tag])

            print("\n##################expert_token_nums result##################")
            print("expert_token_nums: ", result_dispatch_ffn_combine)
            all_result = all(result_dispatch_ffn_combine.values())
            compare_result = "Success" if all_result else "Failed"
            print(f"continue test {compare_result} ---------------------- 用例行数为{data_line + 1}，ID为{id_part}")
            
        except Exception as e:
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            write_row_to_csv('moe_distribute.csv', [data_line + 1, current_case_name, f"x_{rank}", f"('FAIL', 异常)", time_str, args.tag])
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"[ERROR] Case FAILED and SKIPPED: Line {data_line + 1}, ID {id_part}")
            print(f"[ERROR] Reason: {str(e)}")
            print(f"[ERROR] Stack Trace:\n{traceback.format_exc()}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")  

        with lock:
            shared_counters[tmp].value += 1
            print(f"[rank{rank}] counter{tmp} is {shared_counters[tmp].value}",flush = True)
        # start_time = time.time()
        while True:
            with lock:
                if shared_counters[tmp].value == rank_per_dev:
                    break
            # current_time = time.time()
            # elapsed_time = current_time - start_time
            # if elapsed_time >= 500:
            #     print(f"[ERROR] rank{rank} timeout")
            #     break
        tmp += 1
    dist.destroy_process_group() # 遍历结束，手动结束通信域

def compare_output_triangle(compared_output: ComparedOutput, cascade_per_card, golden_per_card, npu_per_card, compare_dtype, compare_per_rank, rank):
    def compare(cascade, golden, npu, dtype, tensor_name, diffThd=0.001, pctThd=0.001):
        np_output_npu = np.array(npu.cpu())
        np_output_golden = np.array(golden.cpu())
        np_output_cascade = np.array(cascade.cpu()) if cascade != 'two comparison' else cascade

        max_diff_hd = 0.0
        if dtype == torch.int8:
            max_diff_hd = 1.0
        if isinstance(np_output_cascade, str) and np_output_cascade == 'two comparison':
            res_output = toolss_pro.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)
            print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output, time_str, args.tag])
            return res_output[0] == 'PASS'

        else:
            res_output = toolss_pro.data_compare_triangle(np_output_cascade, np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd,
                                            max_diff_hd=max_diff_hd)
            return None
        # print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
        # timestamp = time.time()
        # # 转换为本地时间结构体，再格式化为秒级字符串
        # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        # write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output, time_str])
        # return res_output[0] == 'PASS'

    mark_str = compared_output.value.upper().replace('_', ' ')
    result = {}

    if not compare_per_rank:
        print(f"##################Compare {mark_str}##################")
        res = compare(
            cascade_per_card.to(compare_dtype) if cascade_per_card != 'two comparison' else cascade_per_card,
            golden_per_card.to(compare_dtype),
            npu_per_card.to(compare_dtype),
            compare_dtype,
            compared_output.value
        )
        result[compared_output.value] = res
    else:
        # for i in (range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)):
        print(f"##################Compare {mark_str} in rank, {rank}##################")

        res = compare(
            cascade_per_card.to(compare_dtype) if cascade_per_card != 'two comparison' else cascade_per_card,
            golden_per_card.to(compare_dtype),
            npu_per_card.to(compare_dtype),
            compare_dtype,
            f"{compared_output.value}_{rank}"
        )
        result[f"{compared_output.value}_{rank}"] = res
    return result

# endregion

# region 用例读取
def read_case_info(file_path, sheet_name, row_number, keys=None):
    try:
        xlsx_df = pd.read_excel(file_path, sheet_name=sheet_name).fillna('')

        headers = xlsx_df.columns.tolist()
        if not headers:
            print("错误：Excel文件为空或格式不正确")
            return None

        if keys is None:
            keys = headers
        missing_keys = [key for key in keys if key not in headers]
        if missing_keys:
            print(f"警告：某些列不存在于表头中：{', '.join(missing_keys)}")
        keys = [key for key in keys if key in headers]
        if not keys:
            print("错误：没有有效的列名可供提取")
            return None
        
        if row_number < 1 or row_number > len(xlsx_df):
            print(f"错误：找不到行号 {row_number}（有效行范围：1~{len(xlsx_df)}）")
            return None
        target_row = xlsx_df.iloc[row_number - 1].to_dict()
        
        return {key: target_row[key] for key in keys}
            
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    return None
    
# endregion

if __name__ == "__main__":
    start_time = time.time() # 开始时间
    os.system(f'rm /root/atc_data/* -rf')
    clean_previous_profiling_result()

    parser = argparse.ArgumentParser()
    parser.add_argument("-excel", "--excel", type=str, default=None, help="执行的用例文件")
    parser.add_argument("-sheet", "--sheet", type=str, default=None, help="执行的用例表格")
    parser.add_argument("-s", "--start_line", type=int, default=0, help="执行的用例start id")
    parser.add_argument("-e", "--end_line", type=int, default=0, help="执行的用例end id")
    parser.add_argument("-m", "--run_mode", type=int, default=0, help="0为复用通信域，1是每条case单独起通信域")
    parser.add_argument("-t", "--tag", type=str, default='rdv', help="给结果记录一个tag")
    args = parser.parse_args()
    excel_max_row = len(pd.read_excel(args.excel, sheet_name=args.sheet))
    args.end_line = min(args.end_line, excel_max_row + 1) 
    max_buff_size = 0
    if args.excel and args.sheet and args.start_line and args.end_line:  
        for data_line in range(args.start_line - 1, args.end_line):
            data = read_case_info(args.excel, args.sheet, data_line)
            init_case_info(data)
            buff_size = npu_get_mega_moe_ccl_buffer_size(ep_world_size, moe_expert_num, bs, k, h, quant_mode, dispatch_quant_out_type,
                                                comm_quant_mode, comm_alg)
            max_buff_size = max(max_buff_size, buff_size)
            # 生成golden和输入
            case_sub_dir = os.path.join("./input", args.sheet, current_case_name)

            if not os.path.exists(case_sub_dir) or not os.path.isfile(os.path.join(case_sub_dir, "x_active_mask.pt")):
                # os.system(f"rm -rf {case_sub_dir}")
                os.makedirs(case_sub_dir, exist_ok=True)

                # 设置随机种子
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                input_data = GenInputAndGolden(**input_kwargs)
                # 公共输入输出
                torch.save(input_data.x_final, os.path.join(case_sub_dir, f"x_final.pt"))
                torch.save(input_data.w1_npu, os.path.join(case_sub_dir, f"weight1.pt"))
                torch.save(input_data.w2_npu, os.path.join(case_sub_dir, f"weight2.pt"))
                torch.save(input_data.weight_scales1, os.path.join(case_sub_dir, f"weight1_scale.pt"))
                torch.save(input_data.weight_scales2, os.path.join(case_sub_dir, f"weight2_scale.pt"))
                torch.save(input_data.x, os.path.join(case_sub_dir, f"x.pt"))
                torch.save(input_data.expert_ids, os.path.join(case_sub_dir, f"expert_ids.pt"))
                torch.save(input_data.expert_scales, os.path.join(case_sub_dir, f"expert_scales.pt"))
                torch.save(input_data.expert_token_nums, os.path.join(case_sub_dir, f"expert_token_nums.pt"))
                torch.save(input_data.x_active_mask, os.path.join(case_sub_dir, f"x_active_mask.pt"))
                
                print(f"成功构造第{data_line + 1}行{current_case_name}的对应golden", flush = True)

        print(f"{args.start_line} — {args.end_line}行golden已全部生成")
        # 设置 HCCL_BUFFSIZE 大小
        os.environ['HCCL_BUFFSIZE'] = str(max_buff_size)
        print(f'cur max_buff_size: {max_buff_size}')

        # npu计算及精度对比
        if args.run_mode == 0:  
            gen_npu(run_cascade_npu, args.excel, args.sheet, args.start_line, args.end_line)
        else:   # 可以单独起通信域 如需要打印性能数据
            for data_line in range(args.start_line, args.end_line + 1):
                gen_npu(run_cascade_npu, args.excel, args.sheet, data_line, data_line)
                # if is_dispatch and is_performance and graph_type == 0:
                #         dispatch_performance = get_dispatch_performance()    
                # if is_combine and is_performance and graph_type == 0:
                #         combine_performance = get_combine_performance()       
                # if is_continue and is_performance:
                #         continue_performance = get_continue_performance()

        end_time = time.time()  # 结束时间
        print(f"所有用例执行总时间：{round(end_time - start_time, 2)}")
    else:
        print("❌ 错误：缺少核心必选参数！excel、sheet、start_line(-s)、end_line(-e)必须同时指定，缺一不可！")
        sys.exit(1)