import os
import csv
import time
import torch
import socket
import torch_npu
import numpy as np
from torch.multiprocessing import Process, Manager
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from math import ceil
from enum import Enum
import torchair
from torchair.configs.compiler_config import CompilerConfig

import toolss_pro
from graph import CascadeGraphModel, MoeDispatchFFNCombineGraphModel
from gen_input_and_golden import GenInputAndGolden
from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, npu_get_mega_moe_ccl_buffer_size

x_active_mask_dim = 0  # mask的维度，当前支持0,1,2

graph_type = 0  # 0为单算子，1为静态图模式，2为动态图模式, 3为aclgraph
is_graph = (graph_type != 0)
quant_mode = 4  # 0为dispatch算子非量化场景，1为静态量化（当前不支持），2为动态量化
comm_quant_mode = 0  # 0无量化，1则combine通信采用int12动态量化（不支持），2为int8动态量化, 3为e5m2, 4为e4m3
is_independent_group_buffersize = 1
ep_buffersize = 6000
tp_buffersize = 1000

is_performance = 1  # 是否进行性能检测
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
bs = 512
local_moe_expert_num = 4
is_cloud = False  # 是否在云上环境进行测试，使用云上环境时设置为True，使用物理机时设为False
if not is_cloud:
    server_index = 0 if server_num > 1 else 0  # 多机每个脚本保持一致并递增index值
    master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'  # 如果通信不通，查看一下master_ip是否设对
else:
    # 云上环境用下面方式获取server_index和master_ip
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

rank_per_dev = 2  # 每个host有几个die
is_mask = 1 if x_active_mask_dim == 1 else 0
is_unfixed_real_token = is_mask or is_variable_bs  # 每张卡上真实token数是否固定
world_size = server_num * rank_per_dev  # 通信域总die数
ep_world_size = world_size // tp_world_size
shared_expert_rank_num = shared_expert_num * rank_num_per_shared_expert  # 共享专家卡数
moe_rank_num = ep_world_size - shared_expert_rank_num  # moe专家卡数
moe_expert_num = moe_rank_num * local_moe_expert_num  # moe专家数

is_shared = (shared_expert_rank_num > 0)


h = 4096  # 每个token的长度
n = 4096
k = 8
comm_alg = ''

transpose_weight = 1

# 交织开关：True 时按交织布局生成 gmm1 的右矩阵 w1_npu（每 256 列 = 128 gate + 128 up）
# 要求 h 和 n 都 256 对齐。
# 当前 kernel 侧需要把 MegaMoe<...> 的 IsInterleaved 模板参传 true 才会真正按交织读取。
is_interleaved = False

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
swiglu_quant_dtype = "float8_e4m3fn"  # 当前仅支持swiglu_quant_dict表量化模式，swiglu量化模式
quant_dtype = "float8_e4m3fn"    # 当前仅支持quant_dtype_dict表量化模式，dispatch量化模式
weight_dtype = "float4_e2m1"    # 当前仅支持quant_dtype_dict表量化模式，weight量化模式
dispatch_quant_out_type = dispatch_quant_out_type_dict[quant_dtype]
scale_dtype = torch.float8_e8m0fnu # 当前仅支持torch.float8_e8m0fnu
is_topk_balance = True
is_x_random = True  # 是否让x值随机
random_seed = 831  # 随机种子，用于固定各随机输入变量的产生，如x,expert_ids,expert_scales等
expert_scales_dtype = torch.float32
current_case_name = 'MOE_Dis_Com_16p_bs12_k6_xdtype0_quant2_graph0_scale1_cascade0_tp1_localmoe3_share1_moe21_ID034'

ep_ranks_list = [list(range(tp_id, world_size, tp_world_size)) for tp_id in range(tp_world_size)]
tp_ranks_list = [list(range(ep_id * tp_world_size, (ep_id + 1) * tp_world_size)) for ep_id in range(ep_world_size)]
server_ranks = slice(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)

print_compare_details = False

buff_size = npu_get_mega_moe_ccl_buffer_size(ep_world_size, moe_expert_num, bs, k, h, quant_mode, dispatch_quant_out_type,
                                             comm_quant_mode, comm_alg)
os.environ['HCCL_BUFFSIZE'] = f'{buff_size}'

print('======================================================')
print(f'{current_case_name=}')
print(f'{buff_size=}')
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
print(f'{is_interleaved=}')
print('======================================================')

if is_interleaved:
    assert h % 256 == 0, f"is_interleaved=True 要求 h ({h}) 256 对齐"
    assert n % 256 == 0, f"is_interleaved=True 要求 n ({n}) 256 对齐"
    print(f'[INTERLEAVE CHECK] h={h} ({h//256}×256) ✓, n={n} ({n//256}×256) ✓; '
          f'gmm1 右矩阵将按 [128 gate | 128 up] 每 256 列交织')

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


# ============================================================================
# Timestamp CSV 写出（解析逻辑全在 scripts/parse_trace.py 里，独立可用）
# ============================================================================
# Kernel 端把 us 时间戳写入 y2GmAddr 前 (TOTAL_CORES * POINTS_PER_CORE * 8) 字节，
# layout = (globalCoreIdx, pointIdx) row-major：
#   AIC[0..36) 在 row [0..36)
#   AIV[0..72) 在 row [36..108)（AIV0=偶数行, AIV1=奇数行，对应同 pair 的 sub0/sub1）
#
# 本文件只负责"读 y → 写 csv"，每 rank 一个 csv（megamoe_trace_rank<N>_<ts>.csv）。
# 解析在 scripts/parse_trace.py，可以离线对已有 csv 重复跑，不必 re-run kernel。
TRACE_ENABLED = True

# 复用 parse_trace 的 layout 常量 + header 定义（保持单一来源）。
# 若 parse_trace.py 不可用（极少见），退回到本文件内的最小常量。
try:
    import parse_trace as _ptr
    TRACE_POINTS_PER_CORE = _ptr.POINTS_PER_CORE
    TRACE_MAX_AIC = _ptr.MAX_AIC
    TRACE_MAX_AIV = _ptr.MAX_AIV
    TRACE_TOTAL_CORES = _ptr.TOTAL_CORES
    _trace_csv_header = _ptr.trace_csv_header
    _analyze_csv = _ptr.analyze_csv
except ImportError:
    _ptr = None
    TRACE_POINTS_PER_CORE = 260
    TRACE_MAX_AIC = 36
    TRACE_MAX_AIV = 72
    TRACE_TOTAL_CORES = TRACE_MAX_AIC + TRACE_MAX_AIV
    def _trace_csv_header():
        return ['rank', 'side', 'core_idx'] + [f'p{i}' for i in range(TRACE_POINTS_PER_CORE)]
    _analyze_csv = None


def _y_tensor_to_uint64(y_tensor):
    """bf16 张量字节直接 view 成 uint64（torch.uint64 在新版才支持，旧版回退）。"""
    y_cpu = y_tensor.cpu().contiguous()
    try:
        return y_cpu.view(torch.uint64).flatten().numpy()
    except Exception:
        return np.frombuffer(y_cpu.view(torch.uint8).numpy().tobytes(), dtype=np.uint64)


def dump_trace_csv_for_rank(y_tensor, rank_id, csv_path):
    """单 rank 写 csv，覆盖模式（每跑一次生成新文件名所以不会冲突）。
       返回 True 表示成功写入。
    """
    if not TRACE_ENABLED:
        return False
    needed_u64 = TRACE_TOTAL_CORES * TRACE_POINTS_PER_CORE
    arr = _y_tensor_to_uint64(y_tensor)
    if arr.size < needed_u64:
        print(f"[TRACE rank {rank_id}] y too small: have {arr.size} u64, need {needed_u64}; "
              f"set bs >= ceil({needed_u64 * 8} / (h*2)) or shrink POINTS_PER_CORE")
        return False
    ts = arr[:needed_u64].reshape(TRACE_TOTAL_CORES, TRACE_POINTS_PER_CORE)
    header = _trace_csv_header()
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for ci in range(TRACE_TOTAL_CORES):
            if ci < TRACE_MAX_AIC:
                side = 'AIC'
                core_idx = ci
            else:
                aiv_idx = ci - TRACE_MAX_AIC
                side = 'AIV0' if (aiv_idx % 2 == 0) else 'AIV1'
                core_idx = aiv_idx
            row = [rank_id, side, core_idx] + [int(x) for x in ts[ci].tolist()]
            w.writerow(row)
    print(f"[TRACE rank {rank_id}] wrote {csv_path}  ({TRACE_TOTAL_CORES}×{TRACE_POINTS_PER_CORE})")
    return True

# dispatch_profiling_path = './Profiling_Result_mc2_dispatch_16p'
# combine_profiling_path = './Profiling_Result_mc2_combine_16p'
cascade_profiling_path = './Profiling_Result_mc2_cascade_16p'
dispatch_op_type = 'MoeDistributeDispatchV3'
combine_op_type = 'MoeDistributeCombineV3'
matmul_swiglu_op_type = 'GroupedMatmulSwigluQuantV2'
dispatch_ffn_combine_op_type = 'MegaMoe'
path_to_op_types = {
    # dispatch_profiling_path: [dispatch_op_type],
    # combine_profiling_path: [combine_op_type],
    cascade_profiling_path: [dispatch_op_type, combine_op_type, matmul_swiglu_op_type, dispatch_ffn_combine_op_type]
}

input_kwargs = {
    'bs': bs, 'h': h, 'n': n, 'quant_dtype': quant_dtype,
    'is_x_random': is_x_random, 'global_bs': global_bs, 'out_dtype': out_dtype, 'moe_expert_num': moe_expert_num,
    'is_topk_balance': is_topk_balance, 'k': k, 'is_mask': is_mask, 'is_unfixed_real_token': is_unfixed_real_token,
    'ep_world_size': ep_world_size, 'is_variable_bs': is_variable_bs, 'expert_scales_dtype': expert_scales_dtype,
    'local_moe_expert_num': local_moe_expert_num, 'quant_mode': quant_mode, 'transpose_weight': transpose_weight,
    'comm_quant_mode': comm_quant_mode, 'weight_dtype': weight_dtype, 'swiglu_quant_dtype': swiglu_quant_dtype,
    'is_interleaved': is_interleaved,
}

cascade_graph_kwargs = {
    'loop': loop, 'is_performance': is_performance, "aic_core_num": aic_core_num, "aiv_core_num": aiv_core_num,
    "is_graph": is_graph
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

def get_cascade_kwargs(rank, input_data, ep_hcomm_info, tp_hcomm_info):
    if weight_dtype == "float4_e2m1" or weight_dtype == "float4_e1m2":
        weight1 = input_data.w1_npu[rank]
        weight1 = torch.from_numpy(pack_float4_to_uint8(weight1))
        weight2 = input_data.w2_npu[rank]
        weight2 = torch.from_numpy(pack_float4_to_uint8(weight2))
    else:
        weight1 = input_data.w1_npu[rank].tobytes()
        weight1 = torch.frombuffer(weight1, dtype=quant_dtype_dict[quant_dtype]).reshape(input_data.w1_npu[rank].shape).clone()
        weight2 = input_data.w2_npu[rank].tobytes()
        weight2 = torch.frombuffer(weight2, dtype=quant_dtype_dict[quant_dtype]).reshape(input_data.w2_npu[rank].shape).clone()
    weight1_scale = input_data.weight_scales1[rank].tobytes()
    weight1_scale = torch.frombuffer(weight1_scale, dtype=torch.float8_e8m0fnu).reshape(input_data.weight_scales1[rank].shape).clone()
    weight2_scale = input_data.weight_scales2[rank].tobytes()
    weight2_scale = torch.frombuffer(weight2_scale, dtype=torch.float8_e8m0fnu).reshape(input_data.weight_scales2[rank].shape).clone()
    return [
        input_data.x[rank].npu(), input_data.expert_ids[rank].npu(), input_data.expert_scales[rank].npu(),
        input_data.x_active_mask[rank].npu(), ep_hcomm_info, tp_hcomm_info, rank, ep_world_size, moe_expert_num, quant_mode,
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

def get_dispatch_ffn_combine_kwargs(rank, input_data, distribute_buffer):
    if quant_dtype == "float4_e2m1" or quant_dtype == "float4_e1m2":
        weight1 = input_data.w1_npu[rank]
        weight1 = torch.from_numpy(pack_float4_to_uint8(weight1))
        weight2 = input_data.w2_npu[rank]
        weight2 = torch.from_numpy(pack_float4_to_uint8(weight2))
    else:
        weight1 = input_data.w1_npu[rank].tobytes()
        weight1 = torch.frombuffer(weight1, dtype=quant_dtype_dict[quant_dtype]).reshape(input_data.w1_npu[rank].shape).clone()
        weight2 = input_data.w2_npu[rank].tobytes()
        weight2 = torch.frombuffer(weight2, dtype=quant_dtype_dict[quant_dtype]).reshape(input_data.w2_npu[rank].shape).clone()
    weight1_scale = input_data.weight_scales1[rank].tobytes()
    weight1_scale = torch.frombuffer(weight1_scale, dtype=torch.float8_e8m0fnu).reshape(input_data.weight_scales1[rank].shape)
    weight2_scale = input_data.weight_scales2[rank].tobytes()
    weight2_scale = torch.frombuffer(weight2_scale, dtype=torch.float8_e8m0fnu).reshape(input_data.weight_scales2[rank].shape)
    return {
        "x": input_data.x[rank].npu(),
        "topk_ids": input_data.expert_ids[rank].npu(),
        "topk_weights": input_data.expert_scales[rank].npu(),
        "l1_weights": weight1.npu(),
        "l2_weights": weight2.npu(),
        "sym_buffer": distribute_buffer,
        "l1_weights_sf": None if quant_mode==0 else weight1_scale.npu(),
        "l2_weights_sf": None if quant_mode==0 else weight2_scale.npu(),
        "x_active_mask": input_data.x_active_mask[rank].npu(),
        "weight1_type": dispatch_quant_out_type_dict[weight_dtype],
        "weight2_type": dispatch_quant_out_type_dict[weight_dtype],
        "rank": rank,
    }
    

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

def run_cascade_npu(queue, rank, input_data):
    set_device(rank)
    # _, _, group_ep, _ = init_hccl_comm(rank)
    ep_hcomm_info_dispatch, tp_hcomm_info_dispatch, _, _, _, ep_group_mega = init_hccl_comm_two_ep(rank)


    cascade_kwargs = get_cascade_kwargs(rank, input_data, ep_hcomm_info_dispatch, tp_hcomm_info_dispatch)
    distribute_buffer = get_symm_buffer_for_mega_moe(**get_symm_buffer_kwargs(ep_group_mega))
    dispatch_ffn_combine_kwargs = get_dispatch_ffn_combine_kwargs(rank, input_data, distribute_buffer)
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
    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    # 获得x中的有效token
    if is_mask:
        real_tokens = input_data.x_active_mask.cpu().sum().item()
        x = x[:real_tokens, :]
        y = y[:real_tokens, :]
    queue.put([
        rank,
        [x.cpu(), y.cpu(), dispatch_out[3].cpu(), expert_token_nums.cpu()]
        # [x.cpu(), x.cpu(), dispatch_out[3].cpu(), dispatch_out[3].cpu()]
        # [x.cpu(), x.cpu(), dispatch_out[3].cpu(), dispatch_out[3].cpu()]
        # [y.cpu(), y.cpu(), expert_token_nums.cpu(), expert_token_nums.cpu()]
    ])

def compare_output_triangle(compared_output: ComparedOutput, cascade_per_card, golden_per_card, npu_per_card, compare_dtype, compare_per_rank):
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
            write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output])
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
            torch.cat(cascade_per_card).to(compare_dtype) if cascade_per_card != 'two comparison' else cascade_per_card,
            torch.cat(golden_per_card).to(compare_dtype),
            torch.cat(npu_per_card).to(compare_dtype),
            compare_dtype,
            compared_output.value
        )
        result[compared_output.value] = res
    else:
        for i in (range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)):
            print(f"##################Compare {mark_str} in rank", i, "epId", i // tp_world_size, "tpId",
                  i % tp_world_size, "##################")
            local_rank_id = i - server_index * rank_per_dev
            print(f'local_rank_id2374:{local_rank_id}')

            res = compare(
                cascade_per_card[local_rank_id].to(compare_dtype) if cascade_per_card != 'two comparison' else cascade_per_card,
                golden_per_card[local_rank_id].to(compare_dtype),
                npu_per_card[local_rank_id].to(compare_dtype),
                compare_dtype,
                f"{compared_output.value}_{i}"
            )
            result[f"{compared_output.value}_{i}"] = res

    return result

if __name__ == "__main__":
    # 设置随机种子
    check_input()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    input_data = GenInputAndGolden(**input_kwargs)

    if int(server_num) > 1:
        print("multi_server scene!!!!! only support running case one by one!!!!!!!!!!!!!")
        rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    else:
        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
    
    print(f"rank list is: {rank_list}")
    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("spawn", force=True)
    for rank in rank_list:
        proc = Process(target=run_cascade_npu, kwargs={"queue": result_queue, "rank": rank, "input_data": input_data})
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

    # ====== 打点：每 rank 一个 csv；解析委托给 parse_trace.py ======
    if TRACE_ENABLED:
        trace_stamp = time.strftime('%Y%m%d_%H%M%S')
        csv_paths = []
        for local_rank_id in range(rank_per_dev):
            global_rank_id = server_index * rank_per_dev + local_rank_id
            csv_path = f'megamoe_trace_rank{global_rank_id}_{trace_stamp}.csv'
            try:
                if dump_trace_csv_for_rank(category_outputs[1][local_rank_id],
                                           global_rank_id, csv_path):
                    csv_paths.append(csv_path)
            except Exception as e:
                print(f"[TRACE rank {global_rank_id}] dump failed: {e}")
        # 立即跑一遍解析（如果 parse_trace 可用）
        if _analyze_csv is not None:
            for p in csv_paths:
                try:
                    _analyze_csv(p)
                except Exception as e:
                    print(f"[TRACE] parse {p} failed: {e}")
        else:
            print(f"[TRACE] parse_trace.py 不可用；离线解析请运行:")
            if csv_paths:
                print(f"        python parse_trace.py {' '.join(csv_paths)}")

    result_dispatch_ffn_combine = {}
    # 比较y
    golden_x = input_data.x_final

    res = compare_output_triangle(
        ComparedOutput.X,
        category_outputs[0],
        golden_x[server_ranks],
        category_outputs[1],
        torch.float32,
        True,
    )
    # result_combine.update(res)

    # 比较expert_token_num
    golden_expert_token_nums = input_data.expert_token_nums

    res = compare_output_triangle(
        ComparedOutput.EXPERT_TOKEN_NUMS,
        'two comparison',
        golden_expert_token_nums[server_ranks],
        category_outputs[3],
        torch.float32,
        False,
    )
    result_dispatch_ffn_combine.update(res)
    for i in range(rank_per_dev):
        print("##################Compare x in rank",i,"epId",i//tp_world_size,"tpId",i%tp_world_size,"##################")

        rst_npu_x = toolss_pro.checkResult(category_outputs[1][i], golden_x[server_ranks][i], "npu_x")
        if print_compare_details:
            rst_npu_x.print_result()
        rst_npu_golden_x = toolss_pro.checkResult(category_outputs[0][i], golden_x[server_ranks][i], "npu_golden_x")
        if print_compare_details:
            rst_npu_golden_x.print_result()
        str1, str2 = rst_npu_x.check_result(rst_npu_golden_x)
        if 'error' in str1:
            res = 'FAIL'
        else:
            res = 'PASS'
        print('[INFO] tensor x output 精度结果为：{}'.format(res))
        write_row_to_csv('moe_combineARN_triCompare.csv', [current_case_name, f"output_x_rank{i}", res])

    print("\n##################expert_token_nums result##################")
    print("expert_token_nums: ", result_dispatch_ffn_combine)
    all_result = all(result_dispatch_ffn_combine.values())
    if is_performance:
        cascade_performance = get_cascade_performance()