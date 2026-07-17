import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Parse output settings before imports that can trigger JIT compilation.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("-o", "--output_dir", type=str, default=".")
_pre_args, _ = _pre_parser.parse_known_args()
_output_dir = os.path.abspath(_pre_args.output_dir)
os.makedirs(_output_dir, exist_ok=True)
_cache_dir = os.path.join(_output_dir, "torch_extensions_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = _cache_dir

# Forkserver children execute module-level code only, so initialize these paths here.
cascade_profiling_path = os.path.join(_output_dir, 'Profiling_Result_mc2_cascade_16p')
csv_result_dir = os.path.join(_output_dir, 'csv_result')

import csv
import copy
import json
import time
import unicodedata
import torch
import socket
import torch_npu
import traceback
import numpy as np
from torch.multiprocessing import Process, Manager, Lock, Value
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from enum import Enum
import torchair
from torchair.configs.compiler_config import CompilerConfig

import toolss_pro
from graph import CascadeGraphModel as PrefillCascadeGraphModel, MegaMoeGraphModel
from graph_decode import CascadeGraphModel as DecodeCascadeGraphModel
from gen_input_and_golden import GenInputAndGolden, iter_rank_npu_weight_chunks, mx_array_to_torch
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe


def get_cascade_graph_model_class():
    return PrefillCascadeGraphModel if moe_stage == "prefill" else DecodeCascadeGraphModel

GOLDEN_FFN_CHUNK_SIZE = 16
GOLDEN_LOG_INTERVAL = 512
GOLDEN_LOG_STATS = False
GOLDEN_BATCH_MATMUL = False
GOLDEN_BATCH_MATMUL_MIN_M = 128
GOLDEN_TORCH_NUM_INTEROP_THREADS = 1
GOLDEN_METADATA_FILE = "metadata.json"
GOLDEN_REQUIRED_FILES = (
    "x_final.pt",
    "x.pt",
    "expert_ids.pt",
    "expert_scales.pt",
    "expert_token_nums.pt",
    "x_active_mask.pt",
    "topk_weights_type.pt",
)
PERSISTED_WEIGHT_FILES = (
    "weight1.pt",
    "weight2.pt",
    "weight1_scale.pt",
    "weight2_scale.pt",
    "shared_weight1.pt",
    "shared_weight2.pt",
    "shared_weight1_scale.pt",
    "shared_weight2_scale.pt",
)
_torch_interop_threads_configured = False
_xlsx_cache = {}


def get_total_cpu_count():
    return max(1, os.cpu_count() or 1)


def get_golden_parallel_config(chunk_size=GOLDEN_FFN_CHUNK_SIZE):
    cpu_count = get_total_cpu_count()
    expert_workers = min(chunk_size, max(1, cpu_count // 4))
    torch_num_threads = max(1, (cpu_count + expert_workers - 1) // expert_workers)
    return cpu_count, expert_workers, torch_num_threads


def set_torch_cpu_threads(num_threads, num_interop_threads):
    global _torch_interop_threads_configured
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if num_interop_threads > 0 and not _torch_interop_threads_configured:
        try:
            if torch.get_num_interop_threads() != num_interop_threads:
                torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError:
            pass
        _torch_interop_threads_configured = True


REQUIRED_EXCEL_COLUMNS = (
    "case_name", "bs", "h", "n", "k", "graph_type", "quant_mode",
    "quant_mode_type", "swiglu_quant_dtype", "weight_dtype", "weight_scale1",
    "weight_scale2", "ep_world_size", "expert_per_rank", "moe_expert_num",
    "seed", "is_performance", "topk_weights_type", "moe_stage",
    "combine_quant_mode", "local_moe_expert_num", "max_recv_token_num",
    "clamp_limit", "loop_acc_time", "A8W8_is_NZ",
)


def require_excel_value(data, key):
    if key not in data:
        raise KeyError(f"Excel missing required column: {key}")
    value = data[key]
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("none", "null", "nan"):
            raise ValueError(f"Excel required value is empty: {key}")
    elif value is None or pd.isna(value):
        raise ValueError(f"Excel required value is empty: {key}")
    return value


def parse_required_int(data, key):
    return int(float(require_excel_value(data, key)))


def parse_required_float(data, key):
    return float(require_excel_value(data, key))


def parse_required_string(data, key):
    return str(require_excel_value(data, key)).strip()


def validate_required_excel_data(data):
    for key in REQUIRED_EXCEL_COLUMNS:
        require_excel_value(data, key)

def init_case_info(data):
    global graph_type, moe_stage, quant_mode, comm_quant_mode, is_quant, is_performance, loop_acc_time, loop, repeat, warmup
    global is_variable_bs, is_full_core, is_topk_balance, is_x_random, random_seed, is_mask, is_unfixed_real_token
    global x_active_mask_dim, server_num, tp_world_size, rank_per_dev, world_size, ep_world_size
    global server_index, master_ip, host_url_list, ep_ranks_list, tp_ranks_list, expert_per_rank, shared_expert_num
    global local_moe_expert_num, moe_expert_num
    global core_num, aic_core_num, aiv_core_num, is_independent_group_buffersize
    global ep_buffersize, tp_buffersize, bs, h, n, k, global_bs, A_moe
    global max_recv_token_num
    global out_dtype, current_case_name, quant_dtype_dict, swiglu_quant_dict, dispatch_quant_out_type_dict
    global swiglu_quant_dtype, quant_dtype, weight_dtype, dispatch_quant_out_type, scale_dtype, clamp_limit
    global A8W8_is_NZ
    global topk_weights_type
    global expert_scales_dtype, transpose_weight, is_graph, comm_alg, is_cloud, server_ranks
    global golden_ffn_chunk_size, fast_random_weight, golden_batch_matmul, golden_batch_matmul_min_m
    global golden_expert_workers, golden_log_interval, golden_log_stats, golden_cpu_count
    global golden_torch_num_threads, golden_torch_num_interop_threads
    global print_compare_details, buff_size, input_kwargs, cascade_graph_kwargs

    validate_required_excel_data(data)
    weight_scale1 = parse_required_string(data, 'weight_scale1')
    weight_scale2 = parse_required_string(data, 'weight_scale2')
    excel_moe_expert_num = parse_required_int(data, 'moe_expert_num')
    # region 控制参数配置

    # region 执行哪些测试

    x_active_mask_dim = 0  # mask的维度，当前支持0,1,2

    graph_type = parse_required_int(data, 'graph_type')  # 0为单算子，1为静态图模式，2为动态图模式, 3为aclgraph
    moe_stage = parse_required_string(data, 'moe_stage').lower()
    is_graph = (graph_type != 0)
    quant_mode = parse_required_int(data, 'quant_mode')  # 0为dispatch算子非量化场景，1为静态量化（当前不支持），2为动态量化
    comm_quant_mode = parse_required_int(data, 'combine_quant_mode')  # 0无量化，3为e5m2, 4为e4m3
    if comm_quant_mode not in (0, 3, 4):
        raise ValueError(f"combine_quant_mode only supports 0, 3, 4, got {comm_quant_mode}")
    is_independent_group_buffersize = 1
    ep_buffersize = 6000
    tp_buffersize = 1000

    is_performance = parse_required_int(data, 'is_performance') # 是否进行性能检测
    loop_acc_time = parse_required_int(data, 'loop_acc_time')
    if loop_acc_time < 1:
        raise ValueError(f"loop_acc_time must be >= 1, got {loop_acc_time}")
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
    expert_per_rank = parse_required_int(data, 'expert_per_rank')

    # 赋值给原来的变量名，后面的代码无需改动
    bs = parse_required_int(data, 'bs')
    local_moe_expert_num = parse_required_int(data, 'local_moe_expert_num')
    if local_moe_expert_num > expert_per_rank:
        raise ValueError(
            f"local_moe_expert_num should be <= expert_per_rank, got "
            f"{local_moe_expert_num} > {expert_per_rank}"
        )
    shared_expert_num = expert_per_rank - local_moe_expert_num
    is_cloud = False  # 是否在云上环境进行测试，使用云上环境时设置为True，使用物理机时设为False
    if not is_cloud:
        server_index = 0 if server_num > 1 else 0  # 多机每个脚本保持一致并递增index值
        master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'  # 如果通信不通，查看一下master_ip是否设对
    else:
        # 云上环境用下面方式获取server_index和master_ip
        server_index = int(os.environ.get("VC_TASK_INDEX"))
        host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
        master_ip = socket.gethostbyname(host_url_list[0])

    rank_per_dev = parse_required_int(data, 'ep_world_size')  # 每个host有几个die
    is_mask = 1 if x_active_mask_dim == 1 else 0
    is_unfixed_real_token = is_mask or is_variable_bs  # 每张卡上真实token数是否固定
    world_size = server_num * rank_per_dev  # 通信域总die数
    ep_world_size = world_size // tp_world_size
    moe_expert_num = ep_world_size * local_moe_expert_num  # moe专家数

    h = parse_required_int(data, 'h')  # 每个token的长度
    n = parse_required_int(data, 'n')
    k = parse_required_int(data, 'k')
    comm_alg = ''

    transpose_weight = 1

    global_bs = bs * ep_world_size
    A_moe = global_bs * min(local_moe_expert_num, k)
    max_recv_token_num = parse_required_int(data, 'max_recv_token_num')

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
    swiglu_quant_dtype = parse_required_string(data, 'swiglu_quant_dtype')  # 当前仅支持swiglu_quant_dict表量化模式，swiglu量化模式
    quant_dtype = parse_required_string(data, 'quant_mode_type')    # 当前仅支持quant_dtype_dict表量化模式，dispatch量化模式
    weight_dtype = parse_required_string(data, 'weight_dtype')   # 当前仅支持quant_dtype_dict表量化模式，weight量化模式
    A8W8_is_NZ = parse_required_int(data, 'A8W8_is_NZ') != 0
    clamp_limit = parse_required_float(data, 'clamp_limit')
    topk_weights_type = parse_required_int(data, 'topk_weights_type') != 0
    dispatch_quant_out_type = dispatch_quant_out_type_dict[quant_dtype]
    scale_dtype = torch.float8_e8m0fnu # 当前仅支持torch.float8_e8m0fnu
    is_topk_balance = True
    is_x_random = True  # 是否让x值随机
    random_seed = parse_required_int(data, 'seed')  # 随机种子，用于固定各随机输入变量的产生，如x,expert_ids,expert_scales等
    expert_scales_dtype = torch.float32
    current_case_name = parse_required_string(data, 'case_name')
    golden_ffn_chunk_size = GOLDEN_FFN_CHUNK_SIZE
    fast_random_weight = True
    golden_batch_matmul = GOLDEN_BATCH_MATMUL
    golden_batch_matmul_min_m = GOLDEN_BATCH_MATMUL_MIN_M
    golden_log_interval = GOLDEN_LOG_INTERVAL
    golden_log_stats = GOLDEN_LOG_STATS
    golden_cpu_count, golden_expert_workers, golden_torch_num_threads = get_golden_parallel_config(golden_ffn_chunk_size)
    golden_torch_num_interop_threads = GOLDEN_TORCH_NUM_INTEROP_THREADS
    set_torch_cpu_threads(golden_torch_num_threads, golden_torch_num_interop_threads)

    ep_ranks_list = [list(range(tp_id, world_size, tp_world_size)) for tp_id in range(tp_world_size)]
    tp_ranks_list = [list(range(ep_id * tp_world_size, (ep_id + 1) * tp_world_size)) for ep_id in range(ep_world_size)]
    server_ranks = slice(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)

    print_compare_details = False

    # buff_size = npu_get_mega_moe_ccl_buffer_size(ep_world_size, moe_expert_num, bs, k, h, quant_mode, dispatch_quant_out_type,
    #                                             comm_quant_mode, comm_alg)
    # os.environ['HCCL_BUFFSIZE'] = f'{buff_size}'

    print('======================================================')
    print(f'case_name={current_case_name}')
    # print(f'{buff_size=}')
    print(f'{bs=}')
    print(f'{h=}')
    print(f'{n=}')
    print(f'{k=}')
    print(f'{max_recv_token_num=}')
    print(f'{ep_world_size=}')
    print(f'{expert_per_rank=}')
    print(f'{local_moe_expert_num=}')
    print(f'{shared_expert_num=}')
    print(f'moe_expert_num_excel={excel_moe_expert_num} (unused)')
    print(f'{moe_expert_num=}')
    print(f'seed={random_seed}')
    print(f'{dispatch_quant_out_type=}')
    print(f'{expert_scales_dtype=}')
    print(f'quant_mode_type={quant_dtype}')
    print(f'{out_dtype=}')
    print(f'{transpose_weight=}')
    print(f'{is_performance=}')
    print(f'{loop_acc_time=}')
    print(f'{moe_stage=}')
    print(f'{graph_type=}')
    print(f'{quant_mode=}')
    print(f'combine_quant_mode={comm_quant_mode}')
    print(f'{swiglu_quant_dtype=}')
    print(f'{weight_dtype=}')
    print(f'{A8W8_is_NZ=}')
    print(f'weight_scale1={weight_scale1} (unused)')
    print(f'weight_scale2={weight_scale2} (unused)')
    print(f'{clamp_limit=}')
    print(f'{topk_weights_type=}')
    print(f'{golden_ffn_chunk_size=}')
    print(f'{fast_random_weight=}')
    print(f'{golden_batch_matmul=}')
    print(f'{golden_batch_matmul_min_m=}')
    print(f'{golden_cpu_count=}')
    print(f'{golden_expert_workers=}')
    print(f'{golden_log_interval=}')
    print(f'{golden_log_stats=}')
    print(f'{golden_torch_num_threads=}')
    print(f'{golden_torch_num_interop_threads=}')
    print('======================================================')
    input_kwargs = {
        'bs': bs, 'h': h, 'n': n, 'quant_dtype': quant_dtype,
        'is_x_random': is_x_random, 'global_bs': global_bs, 'out_dtype': out_dtype, 'moe_expert_num': moe_expert_num,
        'is_topk_balance': is_topk_balance, 'k': k, 'is_mask': is_mask, 'is_unfixed_real_token': is_unfixed_real_token,
        'ep_world_size': ep_world_size, 'is_variable_bs': is_variable_bs, 'expert_scales_dtype': expert_scales_dtype,
        'local_moe_expert_num': local_moe_expert_num, 'expert_per_rank': expert_per_rank,
        'shared_expert_num': shared_expert_num, 'quant_mode': quant_mode, 'transpose_weight': transpose_weight,
        'comm_quant_mode': comm_quant_mode, 'weight_dtype': weight_dtype, 'swiglu_quant_dtype': swiglu_quant_dtype,
        'clamp_limit': clamp_limit, 'golden_ffn_chunk_size': golden_ffn_chunk_size,
        'fast_random_weight': fast_random_weight, 'golden_batch_matmul': golden_batch_matmul,
        'golden_batch_matmul_min_m': golden_batch_matmul_min_m,
        'golden_expert_workers': golden_expert_workers, 'golden_log_interval': golden_log_interval,
        'golden_log_stats': golden_log_stats, 'topk_weights_type': topk_weights_type,
        'retain_npu_weights': False,
        'random_seed': random_seed,
    }

    cascade_graph_kwargs = {
        'loop': loop, 'is_performance': is_performance, "aic_core_num": aic_core_num, "aiv_core_num": aiv_core_num,
        "is_graph": is_graph, "graph_type": graph_type, "clamp_limit": clamp_limit,
        "local_moe_expert_num": local_moe_expert_num, "shared_expert_num": shared_expert_num,
        "expert_per_rank": expert_per_rank,
        "topk_weights_type": topk_weights_type,
    }
    check_input()

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

# Keep the existing name for callers while routing all CSV output to output_dir.
precision_result_dir = csv_result_dir
mega_moe_op_type = 'MegaMoe'
small_ops_wall_op_type = 'small_ops_wall'
small_ops_resource_op_type = 'small_ops_resource'
communication_op_type = 'communication'
moe_expert_op_type = 'moe_expert'
shared_expert_op_type = 'shared_expert'
add_op_type = 'Add'
path_to_op_types = {
    cascade_profiling_path: [
        small_ops_wall_op_type, small_ops_resource_op_type,
        communication_op_type, moe_expert_op_type, shared_expert_op_type,
        add_op_type, mega_moe_op_type
    ]
}

def write_row_to_csv(csv_name: str, row_content: list):
    os.makedirs(precision_result_dir, exist_ok=True)
    with open(os.path.join(precision_result_dir, csv_name), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)

def check_input():
    if moe_stage not in ("prefill", "decode"):
        raise ValueError(f"moe_stage only supports prefill or decode, got {moe_stage}")

    if shared_expert_num < 0:
        raise ValueError(
            f"shared_expert_num must be >= 0, got {shared_expert_num}; "
            f"expert_per_rank={expert_per_rank}, local_moe_expert_num={local_moe_expert_num}"
        )

    if moe_stage == "decode" and shared_expert_num > 0:
        raise ValueError("decode stage does not support shared experts")

    if graph_type == 0 and is_performance:
        print('单算子模式(graph_type=0)不支持性能采集，请使用图模式进行性能采集!')
        exit(0)

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
    profiling_result = os.path.join(os.path.dirname(cascade_profiling_path), 'Profiling_Result_')
    os.system(f'rm -rf {profiling_result}*')

PERFORMANCE_TAIL_SIZE = 50
PERFORMANCE_TAIL_SKIP = 2

TIMELINE_METRIC_BY_OP = {
    small_ops_wall_op_type: "wall_span_us",
    small_ops_resource_op_type: "resource_sum_us",
    communication_op_type: "communication_us",
    moe_expert_op_type: "moe_expert_us",
    shared_expert_op_type: "shared_expert_us",
    add_op_type: "add_us",
    mega_moe_op_type: "duration_us",
}

OP_DISPLAY_NAMES = {
    small_ops_wall_op_type: "小算子真实耗时(wall)",
    small_ops_resource_op_type: "小算子耗时(sum)",
    communication_op_type: "通信耗时",
    moe_expert_op_type: "MOE专家耗时",
    shared_expert_op_type: "共享专家耗时",
    add_op_type: "Add耗时",
    mega_moe_op_type: "融合MegaMoe",
}


def _to_float_series(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _text_has(row, *keywords):
    text = f"{row.get('Type', '')} {row.get('Name', '')}".lower()
    return any(keyword.lower() in text for keyword in keywords)


def _read_kernel_details(path):
    kernel_details = pd.read_csv(path)
    kernel_details.rename(columns=lambda column: str(column).strip(), inplace=True)
    required_columns = {"Type", "Start Time(us)", "Duration(us)"}
    if not required_columns.issubset(kernel_details.columns):
        return pd.DataFrame()
    kernel_details["start_us"] = _to_float_series(kernel_details["Start Time(us)"])
    kernel_details["duration_us"] = _to_float_series(kernel_details["Duration(us)"])
    kernel_details["end_us"] = kernel_details["start_us"] + kernel_details["duration_us"]
    if "Stream ID" in kernel_details.columns:
        kernel_details["stream_id"] = pd.to_numeric(kernel_details["Stream ID"], errors="coerce").fillna(-1).astype(int)
    else:
        kernel_details["stream_id"] = -1
    if "Device_id" in kernel_details.columns:
        kernel_details["local_rank_id"] = pd.to_numeric(kernel_details["Device_id"], errors="coerce").fillna(-1).astype(int)
    else:
        kernel_details["local_rank_id"] = -1
    return kernel_details.dropna(subset=["start_us", "duration_us", "end_us"])


def _get_kernel_detail_paths(profiling_path):
    kernel_detail_paths = []
    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("kernel_details") and filename.endswith(".csv"):
                kernel_detail_paths.append(os.path.join(dirpath, filename))
    return sorted(kernel_detail_paths)


def _get_moe_streams(kernel_details):
    route_mask = kernel_details.apply(lambda row: _text_has(row, "MoeInitRouting", "MoeFinalizeRouting"), axis=1)
    return set(kernel_details.loc[route_mask, "stream_id"].dropna().astype(int).tolist())


def _classify_timeline_row(row, moe_streams):
    if _text_has(row, "MegaMoe"):
        return mega_moe_op_type
    if _text_has(row, "allGather", "reduceScatter"):
        return communication_op_type
    if _text_has(row, "Add"):
        return add_op_type
    if _text_has(row, "MoeInitRouting", "MoeFinalizeRouting"):
        return moe_expert_op_type
    if _text_has(row, "GroupedMatmul", "ClippedSwiglu", "DynamicMxQuant", "AntiMxQuant", "DynamicQuant"):
        if moe_stage == "decode":
            return moe_expert_op_type
        return moe_expert_op_type if int(row["stream_id"]) in moe_streams else shared_expert_op_type
    if _text_has(row, "ReduceSum", "Tile", "Fill"):
        return shared_expert_op_type
    if _text_has(row, "MoeDistributeDispatch", "MoeDistributeCombine"):
        return communication_op_type
    return None


def _is_allgather_row(row):
    return row.get("timeline_group") == communication_op_type and _text_has(row, "allGather")


def _is_dispatch_mx_quant_row(row):
    return (
        row.get("timeline_group") == moe_expert_op_type
        and _text_has(row, "DynamicMxQuant")
    )


def _trim_segment_to_real_combine(segment):
    if not (segment["timeline_group"] == shared_expert_op_type).any():
        return segment[segment["timeline_group"] != add_op_type]

    combine_base_mask = segment.apply(
        lambda row: _text_has(row, "reduceScatter", "MoeFinalizeRouting", "MoeDistributeCombine"),
        axis=1,
    )
    if combine_base_mask.any():
        combine_base_indices = segment.index[combine_base_mask].tolist()
        search_start = combine_base_indices[-1]
        add_rows = segment[
            (segment.index > search_start)
            & (segment["timeline_group"] == add_op_type)
        ]
    else:
        add_rows = segment[segment["timeline_group"] == add_op_type]

    if add_rows.empty:
        return segment

    combine_add_end_us = float(add_rows.iloc[0]["end_us"])
    return segment[segment["start_us"] < combine_add_end_us]


def _is_combine_comm_quant_row(row):
    return (
        row.get("timeline_group") == moe_expert_op_type
        and _text_has(row, "DynamicMxQuant", "AntiMxQuant")
    )


def _drop_combine_comm_quant_rows(segment):
    combine_base_mask = segment.apply(
        lambda row: _text_has(row, "MoeFinalizeRouting", "MoeDistributeCombine"),
        axis=1,
    )
    combine_base_indices = segment.index[combine_base_mask].tolist()
    if not combine_base_indices:
        return segment, 0.0

    drop_indices = set()
    for combine_base_index in combine_base_indices:
        gmm_mask = (
            (segment.index < combine_base_index)
            & segment.apply(lambda row: _text_has(row, "GroupedMatmul"), axis=1)
        )
        gmm_indices = segment.index[gmm_mask].tolist()
        if not gmm_indices:
            continue

        reduce_mask = (
            (segment.index > combine_base_index)
            & segment.apply(lambda row: _text_has(row, "reduceScatter", "MoeDistributeCombine"), axis=1)
        )
        reduce_indices = segment.index[reduce_mask].tolist()
        start_index = gmm_indices[-1]
        end_index = reduce_indices[0] if reduce_indices else segment.index[-1] + 1
        quant_mask = (
            (segment.index > start_index)
            & (segment.index < end_index)
            & segment.apply(_is_combine_comm_quant_row, axis=1)
        )
        drop_indices.update(segment.index[quant_mask].tolist())

    if not drop_indices:
        return segment, 0.0

    drop_rows = segment.loc[sorted(drop_indices)]
    excluded_wall_us = float(drop_rows["duration_us"].sum())
    return segment.drop(index=sorted(drop_indices)), excluded_wall_us


def _summarize_rank_metric(rows, metric_name):
    if not rows:
        return [0, 0, 0, 0]
    dataframe = pd.DataFrame(rows)
    if dataframe.empty or metric_name not in dataframe.columns:
        return [0, 0, 0, 0]
    per_rank_medians = []
    per_rank_mins = []
    for _, rank_rows in dataframe.groupby("local_rank_id", dropna=False):
        values = pd.to_numeric(rank_rows[metric_name], errors="coerce").dropna()
        if PERFORMANCE_TAIL_SKIP > 0 and len(values) > PERFORMANCE_TAIL_SKIP:
            values = values.iloc[:-PERFORMANCE_TAIL_SKIP]
        values = values.tail(PERFORMANCE_TAIL_SIZE)
        if values.empty:
            continue
        per_rank_medians.append(float(values.median()))
        per_rank_mins.append(float(values.min()))
    if not per_rank_medians:
        return [0, 0, 0, 0]
    return [min(per_rank_medians), max(per_rank_medians), float(np.median(per_rank_medians)), min(per_rank_mins)]


def _extract_small_op_iterations(kernel_details, kernel_detail_path):
    moe_streams = _get_moe_streams(kernel_details)
    timeline = kernel_details.copy()
    timeline["timeline_group"] = timeline.apply(lambda row: _classify_timeline_row(row, moe_streams), axis=1)
    timeline = timeline[timeline["timeline_group"].notna() & (timeline["timeline_group"] != mega_moe_op_type)]
    timeline = timeline.sort_values(["start_us", "end_us"], kind="stable").reset_index(drop=True)
    if timeline.empty:
        return []
    anchor_positions = [
        index for index, row in timeline.iterrows()
        if _text_has(row, "MoeInitRouting", "MoeDistributeDispatch")
    ]
    if not anchor_positions:
        anchor_positions = [0]
    start_positions = []
    previous_anchor_position = -1
    for anchor_position in anchor_positions:
        search_start = previous_anchor_position + 1
        allgather_positions = [
            index for index in range(search_start, anchor_position)
            if _is_allgather_row(timeline.iloc[index])
        ]
        dispatch_quant_positions = [
            index for index in range(search_start, anchor_position)
            if _is_dispatch_mx_quant_row(timeline.iloc[index])
        ]
        if allgather_positions:
            first_allgather_position = allgather_positions[0]
            quant_before_allgather = [
                index for index in dispatch_quant_positions
                if index < first_allgather_position
            ]
            start_position = (
                quant_before_allgather[-1]
                if quant_before_allgather
                else first_allgather_position
            )
        elif dispatch_quant_positions:
            start_position = dispatch_quant_positions[-1]
        else:
            continue
        if start_positions and start_position <= start_positions[-1]:
            continue
        start_positions.append(start_position)
        previous_anchor_position = anchor_position

    if not start_positions:
        start_positions = [anchor_positions[0]]

    rows = []
    local_rank_id = int(timeline["local_rank_id"].dropna().iloc[0]) if not timeline["local_rank_id"].dropna().empty else -1
    for iteration_id, start_position in enumerate(start_positions):
        end_position = start_positions[iteration_id + 1] if iteration_id + 1 < len(start_positions) else len(timeline)
        segment = timeline.iloc[start_position:end_position]
        if segment.empty:
            continue
        segment = _trim_segment_to_real_combine(segment)
        if segment.empty:
            continue
        segment, excluded_wall_us = _drop_combine_comm_quant_rows(segment)
        if segment.empty:
            continue
        duration_by_group = segment.groupby("timeline_group")["duration_us"].sum().to_dict()
        resource_sum_us = float(segment["duration_us"].sum())
        raw_wall_span_us = float(segment["end_us"].max() - segment["start_us"].min())
        wall_span_us = max(0.0, raw_wall_span_us - excluded_wall_us)
        rows.append({
            "kernel_details_file": kernel_detail_path,
            "local_rank_id": local_rank_id,
            "iteration_id": iteration_id,
            "wall_span_us": wall_span_us,
            "resource_sum_us": resource_sum_us,
            "communication_us": float(duration_by_group.get(communication_op_type, 0.0)),
            "moe_expert_us": float(duration_by_group.get(moe_expert_op_type, 0.0)),
            "shared_expert_us": float(duration_by_group.get(shared_expert_op_type, 0.0)),
            "add_us": float(duration_by_group.get(add_op_type, 0.0)),
            "overlap_saved_us": max(0.0, resource_sum_us - wall_span_us),
        })
    return rows


def _extract_mega_moe_samples(kernel_details, kernel_detail_path):
    rows = []
    mega_rows = kernel_details[kernel_details.apply(lambda row: _text_has(row, "MegaMoe"), axis=1)]
    if mega_rows.empty:
        return rows
    for index, row in mega_rows.sort_values(["start_us", "end_us"], kind="stable").reset_index(drop=True).iterrows():
        rows.append({
            "kernel_details_file": kernel_detail_path,
            "local_rank_id": int(row.get("local_rank_id", -1)),
            "iteration_id": int(index),
            "duration_us": float(row["duration_us"]),
        })
    return rows

def get_performance(profiling_path: str):
    op_types = path_to_op_types[profiling_path]
    kernel_detail_paths = _get_kernel_detail_paths(profiling_path)
    timeline_rows = []
    need_small_ops = any(op_type != mega_moe_op_type for op_type in op_types)
    need_mega_moe = mega_moe_op_type in op_types
    for kernel_detail_path in kernel_detail_paths:
        kernel_details = _read_kernel_details(kernel_detail_path)
        if kernel_details.empty:
            continue
        if need_small_ops:
            timeline_rows.extend(_extract_small_op_iterations(kernel_details, kernel_detail_path))
        if need_mega_moe:
            timeline_rows.extend(_extract_mega_moe_samples(kernel_details, kernel_detail_path))

    if not kernel_detail_paths:
        print(f"[WARN] 未找到 kernel_details.csv: {profiling_path}")

    performance_info = []
    for op_type in op_types:
        metric_name = TIMELINE_METRIC_BY_OP[op_type]
        performance_info.append(_summarize_rank_metric(timeline_rows, metric_name))

    ranks = sorted({row["local_rank_id"] for row in timeline_rows})
    print(
        f"[INFO] {profiling_path} timeline统计: "
        f"kernel_details={len(kernel_detail_paths)}, ranks={ranks}, samples={len(timeline_rows)}"
    )
    return performance_info, None

def get_expected_memory_usage(test_mode: str, rank_id: int):
    def get_dispatch_expected_memory_usage():
        dispatch_output_memory_usage = (
                A_moe * tp_world_size * h * expand_x_dtype_bytes
                + A_moe * tp_world_size * float_bytes
                + A_moe * expand_idx_per_token * int32_bytes
                + local_moe_expert_num * int64_bytes
                + ep_world_size * local_moe_expert_num * int32_bytes
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
        return get_dispatch_expected_memory_usage()

    if test_mode == 'combine':
        return get_combine_expected_memory_usage(ep_rank_id, tp_rank_id)

    if test_mode == 'cascade':
        return max(get_dispatch_expected_memory_usage(),
                   get_combine_expected_memory_usage(ep_rank_id, tp_rank_id))

def format_perf_value(value):
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)

def display_width(value):
    width = 0
    for char in str(value):
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width

def ljust_display(value, width):
    text = str(value)
    return text + " " * max(0, width - display_width(text))

def print_perf_table(title, headers, rows):
    header_row = [str(header) for header in headers]
    table_rows = [[str(item) for item in row] for row in rows]
    widths = [
        max(display_width(header), *(display_width(row[idx]) for row in table_rows))
        for idx, header in enumerate(header_row)
    ]
    print(title)
    print("| " + " | ".join(ljust_display(header, widths[idx]) for idx, header in enumerate(header_row)) + " |")
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in table_rows:
        print("| " + " | ".join(ljust_display(row[idx], widths[idx]) for idx in range(len(header_row))) + " |")

def print_cascade_performance(cascade_performance):
    op_performances, _ = cascade_performance
    op_types = path_to_op_types[cascade_profiling_path]
    perf_rows = []
    for op_type, stats in zip(op_types, op_performances):
        perf_rows.append([
            OP_DISPLAY_NAMES.get(op_type, op_type),
            format_perf_value(stats[3]),
            format_perf_value(stats[0]),
            format_perf_value(stats[1]),
            format_perf_value(stats[2]),
        ])
    print_perf_table(
        '[INFO] MC2性能统计(us)',
        ['统计项', '最小值的最小值', '中位数的最小值', '中位数的最大值', '中位数的中位数'],
        perf_rows,
    )

    stats_by_op = dict(zip(op_types, op_performances))
    small_wall = stats_by_op.get(small_ops_wall_op_type, [0, 0, 0, 0])
    small_resource = stats_by_op.get(small_ops_resource_op_type, [0, 0, 0, 0])
    mega_moe = stats_by_op.get(mega_moe_op_type, [0, 0, 0, 0])
    print_perf_table(
        '[INFO] 性能汇总(us)',
        ['指标', '中位数的中位数'],
        [
            ['小算子真实耗时(wall)', format_perf_value(small_wall[2])],
            ['小算子耗时(sum)', format_perf_value(small_resource[2])],
            ['融合MegaMoe', format_perf_value(mega_moe[2])],
        ],
    )


def get_cascade_performance():
    op_types = path_to_op_types[cascade_profiling_path]
    op_performances, memory_usage = get_performance(cascade_profiling_path)
    print_cascade_performance([op_performances, memory_usage])
    save_performance_result('cascade', current_case_name, op_types, op_performances, memory_usage)
    return [op_performances, memory_usage]

def save_performance_result(test_mode, case_name, op_types, op_performances, op_memory_usage_info):
    def save_execution_time_result(test_mode, case_name, op_types, op_performances):
        csv_name = f'{test_mode}_performance_result.csv'
        title = [
            'op_type', 'case_name',
            'mc2_minOfMedians', 'mc2_maxOfMedians',
            'mc2_medianOfMedians', 'mc2_minOfMins',
            'recordtime'
        ]

        for op_type, op_performance in zip(op_types, op_performances):
            row = [op_type, case_name] + op_performance + [time.strftime("%Y-%m-%d %H:%M:%S")]
            if not os.path.exists(os.path.join(precision_result_dir, csv_name)):
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

        os.makedirs(precision_result_dir, exist_ok=True)
        csv_path = os.path.join(precision_result_dir, csv_name)
        set_header = (not os.path.exists(csv_path))
        df.to_csv(csv_path, header=set_header, index=False, mode='a')

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
        compiled_model = torch.compile(
            model,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options={"frozen_parameter": True})
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

def prepare_float4_packed_weight(weight: np.ndarray) -> torch.Tensor:
    if isinstance(weight, torch.Tensor):
        weight = weight.contiguous().view(torch.uint8)
        weight = weight & 0x0F
        return weight[..., 0::2] | (weight[..., 1::2] << 4)
    return torch.from_numpy(pack_float4_to_uint8(weight))

def prepare_moe_weight(weight, gmm_x_dtype_name):
    weight = weight.npu()
    if weight_dtype == "float4_e2m1":
        if str(gmm_x_dtype_name).lower() == "float4_e2m1":
            return torch_npu.npu_format_cast(weight, 29)
        return torch_npu.npu_format_cast(
            weight,
            29,
            customize_dtype=get_torch_dtype(gmm_x_dtype_name),
            input_dtype=get_torch_dtype(weight_dtype),
        )
    if A8W8_is_NZ and weight_dtype in ("float8_e4m3fn", "float8_e5m2"):
        return torch_npu.npu_format_cast(weight, 29)
    return weight

def restore_weight_input(weight_input, dtype_name):
    if weight_input is None:
        return None
    if weight_dtype == "float4_e2m1":
        return prepare_float4_packed_weight(weight_input)
    return mx_array_to_torch(weight_input, quant_dtype_dict[dtype_name])


def restore_weight_scale_input(weight_scale_input):
    if weight_scale_input is None:
        return None
    return mx_array_to_torch(weight_scale_input, torch.float8_e8m0fnu)


def rebuild_rank_weights_to_npu(ep_id):
    outputs = {
        "weight1": None,
        "weight2": None,
        "weight1_scale": None,
        "weight2_scale": None,
        "shared_weight1": None,
        "shared_weight2": None,
        "shared_weight1_scale": None,
        "shared_weight2_scale": None,
    }
    output_names = {
        "moe": ("weight1", "weight2", "weight1_scale", "weight2_scale"),
        "shared": (
            "shared_weight1",
            "shared_weight2",
            "shared_weight1_scale",
            "shared_weight2_scale",
        ),
    }
    chunk_names = ("w1_npu", "w2_npu", "weight_scales1", "weight_scales2")

    for chunk in iter_rank_npu_weight_chunks(ep_id, **input_kwargs):
        for output_name, chunk_name in zip(output_names[chunk["partition"]], chunk_names):
            if quant_mode == 0 and "scale" in output_name:
                continue
            raw_value = chunk[chunk_name]
            if "scale" in output_name:
                cpu_value = restore_weight_scale_input(raw_value)
            else:
                cpu_value = restore_weight_input(raw_value, quant_dtype)
            if outputs[output_name] is None:
                outputs[output_name] = torch.empty(
                    (chunk["count"],) + tuple(cpu_value.shape[1:]),
                    dtype=cpu_value.dtype,
                    device="npu",
                )
            start = chunk["offset"]
            outputs[output_name][start:start + cpu_value.shape[0]].copy_(cpu_value.npu())
        torch.npu.synchronize()

    outputs["weight1"] = prepare_moe_weight(outputs["weight1"], quant_dtype)
    outputs["weight2"] = prepare_moe_weight(outputs["weight2"], swiglu_quant_dtype)
    if shared_expert_num > 0:
        outputs["shared_weight1"] = prepare_moe_weight(outputs["shared_weight1"], quant_dtype)
        outputs["shared_weight2"] = prepare_moe_weight(outputs["shared_weight2"], swiglu_quant_dtype)
    torch.npu.synchronize()
    return outputs

def combine_cascade_weight_inputs(moe_input, shared_input, dtype_name, is_scale=False):
    restore = restore_weight_scale_input if is_scale else lambda value: restore_weight_input(value, dtype_name)
    moe_value = restore(moe_input)
    if shared_input is None:
        return moe_value
    return torch.cat((restore(shared_input), moe_value), dim=0)

def prepare_cascade_weight_partitions(weight, gmm_x_dtype_name):
    if moe_stage != "prefill":
        return prepare_moe_weight(weight, gmm_x_dtype_name)
    shared_weight = None
    if shared_expert_num > 0:
        shared_weight = prepare_moe_weight(
            weight[:shared_expert_num].contiguous(), gmm_x_dtype_name
        )
    moe_weight = prepare_moe_weight(
        weight[shared_expert_num:expert_per_rank].contiguous(), gmm_x_dtype_name
    )
    return shared_weight, moe_weight

def build_cascade_expert_inputs(rank, expert_ids, expert_scales):
    return expert_ids.contiguous(), expert_scales.contiguous(), moe_expert_num

def should_use_static_mega_moe_weights():
    return graph_type == 3 and (
        weight_dtype == "float4_e2m1"
        or A8W8_is_NZ and weight_dtype in ("float8_e4m3fn", "float8_e5m2")
    )

def prepare_mega_moe_model_for_graph(model, mega_moe_kwargs):
    if should_use_static_mega_moe_weights():
        model.set_static_weights(
            mega_moe_kwargs.pop("l1_weights"),
            mega_moe_kwargs.pop("l2_weights"),
            mega_moe_kwargs.pop("shared_weight1", None),
            mega_moe_kwargs.pop("shared_weight2", None),
        )
    return model

def prepare_cascade_model_for_graph(model, cascade_kwargs):
    if should_use_static_mega_moe_weights():
        model.set_static_weights(cascade_kwargs[11], cascade_kwargs[14])
        cascade_kwargs[11] = None
        cascade_kwargs[14] = None
    return model

def get_cascade_kwargs(
        rank, group_ep, group_tp,
        weight1_input, weight2_input, weight1_scale_input, weight2_scale_input,
        shared_weight1_input, shared_weight2_input,
        shared_weight1_scale_input, shared_weight2_scale_input,
        x, expert_ids, expert_scales, x_active_mask,
        weights_are_prepared=False,
):
    ep_rank_id = rank // tp_world_size
    if weights_are_prepared:
        if moe_stage == "prefill":
            weight1_for_gmm = (shared_weight1_input, weight1_input)
            weight2_for_gmm = (shared_weight2_input, weight2_input)
        else:
            weight1_for_gmm = weight1_input
            weight2_for_gmm = weight2_input
        if shared_weight1_scale_input is None:
            weight1_scale = weight1_scale_input
            weight2_scale = weight2_scale_input
        else:
            weight1_scale = torch.cat((shared_weight1_scale_input, weight1_scale_input), dim=0)
            weight2_scale = torch.cat((shared_weight2_scale_input, weight2_scale_input), dim=0)
    else:
        weight1 = combine_cascade_weight_inputs(weight1_input, shared_weight1_input, quant_dtype)
        weight2 = combine_cascade_weight_inputs(weight2_input, shared_weight2_input, quant_dtype)
        weight1_scale = combine_cascade_weight_inputs(
            weight1_scale_input, shared_weight1_scale_input, quant_dtype, is_scale=True
        )
        weight2_scale = combine_cascade_weight_inputs(
            weight2_scale_input, shared_weight2_scale_input, quant_dtype, is_scale=True
        )
        weight1_for_gmm = prepare_cascade_weight_partitions(weight1, quant_dtype)
        weight2_for_gmm = prepare_cascade_weight_partitions(weight2, swiglu_quant_dtype)
    cascade_expert_ids, cascade_expert_scales, cascade_expert_num = build_cascade_expert_inputs(
        rank, expert_ids, expert_scales
    )
    return [
        x.npu(), cascade_expert_ids.npu(), cascade_expert_scales.npu(),
        x_active_mask.npu(), group_ep, group_tp, ep_rank_id, ep_world_size, cascade_expert_num, quant_mode,
        global_bs, weight1_for_gmm, out_dtype, None if quant_mode==0 else weight1_scale.npu(), weight2_for_gmm,
        None if quant_mode==0 else weight2_scale.npu(), dispatch_quant_out_type, comm_quant_mode, rank,
        get_torch_dtype(weight_dtype) if weight_dtype.lower() in ['float4_e2m1'] else None,
        swiglu_quant_dict[swiglu_quant_dtype],
        get_torch_dtype(quant_dtype) if quant_dtype.lower() in ['float4_e2m1'] else None
    ]

def get_symm_buffer_kwargs(group_ep):
    return {
        "group": group_ep,
        "num_experts": moe_expert_num,
        "num_max_tokens_per_rank": global_bs,
        "num_topk": k,
        "hidden": h,
        "intermediate_hidden": h,
        "max_recv_token_num": max_recv_token_num,
        "dispatch_quant_mode": quant_mode,
        "dispatch_quant_out_dtype": dispatch_quant_out_type,
        "combine_quant_mode": comm_quant_mode,
        # "topk_weights_type": 1 if topk_weights_type else 0,
        "comm_alg": ""
        }

def get_mega_moe_kwargs(
        rank, distribute_buffer,
        weight1_input, weight2_input, weight1_scale_input, weight2_scale_input,
        shared_weight1_input, shared_weight2_input,
        shared_weight1_scale_input, shared_weight2_scale_input,
        expert_ids, x, expert_scales, x_active_mask,
        weights_are_prepared=False,
):
    if weights_are_prepared:
        weight1_for_gmm = weight1_input
        weight2_for_gmm = weight2_input
        weight1_scale = weight1_scale_input
        weight2_scale = weight2_scale_input
    else:
        weight1_for_gmm = prepare_moe_weight(restore_weight_input(weight1_input, quant_dtype), quant_dtype)
        weight2_for_gmm = prepare_moe_weight(restore_weight_input(weight2_input, quant_dtype), swiglu_quant_dtype)
        weight1_scale = restore_weight_scale_input(weight1_scale_input)
        weight2_scale = restore_weight_scale_input(weight2_scale_input)
    mega_moe_kwargs = {
        "x": x.npu(),
        "topk_ids": expert_ids.npu(),
        "topk_weights": expert_scales.npu(),
        "l1_weights": weight1_for_gmm,
        "l2_weights": weight2_for_gmm,
        "sym_buffer": distribute_buffer,
        "l1_weights_sf": None if quant_mode==0 else weight1_scale.npu(),
        "l2_weights_sf": None if quant_mode==0 else weight2_scale.npu(),
        "x_active_mask": x_active_mask.npu(),
        "weight1_type": dispatch_quant_out_type_dict[weight_dtype],
        "weight2_type": dispatch_quant_out_type_dict[weight_dtype],
        "rank": rank,
    }
    if shared_expert_num > 0:
        if weights_are_prepared:
            shared_weight1 = shared_weight1_input
            shared_weight2 = shared_weight2_input
            shared_weight_scales1 = shared_weight1_scale_input
            shared_weight_scales2 = shared_weight2_scale_input
        else:
            shared_weight1 = prepare_moe_weight(
                restore_weight_input(shared_weight1_input, quant_dtype), quant_dtype
            )
            shared_weight2 = prepare_moe_weight(
                restore_weight_input(shared_weight2_input, quant_dtype), swiglu_quant_dtype
            )
            shared_weight_scales1 = restore_weight_scale_input(shared_weight1_scale_input)
            shared_weight_scales2 = restore_weight_scale_input(shared_weight2_scale_input)
        mega_moe_kwargs.update({
            "shared_weight1": shared_weight1,
            "shared_weight2": shared_weight2,
            "shared_weight_scales1": None if quant_mode == 0 else shared_weight_scales1.npu(),
            "shared_weight_scales2": None if quant_mode == 0 else shared_weight_scales2.npu(),
        })
    return mega_moe_kwargs

def gen_npu(target_func, excel, sheet, start_line, end_line):
    def parse_rank_input(
            target_func, rank, excel, sheet, start_line, end_line,
            shared_counters, lock
    ):
        if target_func == run_cascade_npu:
            return {
                "rank": rank,
                "excel": excel,
                "sheet": sheet,
                "start_line": start_line,
                "end_line": end_line,
                "shared_counters": shared_counters,
                "lock":lock,
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
        rank_kwargs = parse_rank_input(
            target_func, rank, excel, sheet, start_line, end_line,
            shared_counters, lock
        )
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

    print(f"device {rank} MegaMoe复用小算子EP域")
    ep_group_mega = ep_group_dispatch

    # Previous path: create a separate EP group for MegaMoe.
    # print(f"device {rank} 初始化EP-2域")
    # for ep2_ranks in ep_ranks_list:
    #     tmp_group = dist.new_group(backend="hccl", ranks=ep2_ranks)
    #     if rank in ep2_ranks:
    #         ep_group_mega = tmp_group

    print(f"device {rank} 初始化TP域")
    for tp_ranks in tp_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=tp_ranks, pg_options = optionstp)
        if rank in tp_ranks:
            tp_group_dispatch = tmp_group

    ep_hcomm_info_dispatch = ep_group_dispatch._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    ep2_hcomm_info_mega = ep_group_mega._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm_info_dispatch = tp_group_dispatch._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    
    return ep_hcomm_info_dispatch, tp_hcomm_info_dispatch, ep2_hcomm_info_mega, ep_group_dispatch, tp_group_dispatch, ep_group_mega


def synchronize_before_mega_moe():
    torch.npu.synchronize()


def run_cascade_npu(
        rank, excel, sheet, start_line, end_line, shared_counters, lock
):
    data = read_case_info(excel, sheet, start_line - 1)
    init_case_info(data)
    set_device(rank)
    _, _, _, ep_group_dispatch, tp_group_dispatch, ep_group_mega = init_hccl_comm_two_ep(rank)
    base_distribute_buffer = get_symm_buffer_for_mega_moe(**get_symm_buffer_kwargs(ep_group_mega))
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
            x = torch.load(os.path.join(case_sub_dir, f"x.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            expert_ids = torch.load(os.path.join(case_sub_dir, f"expert_ids.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            expert_scales = torch.load(os.path.join(case_sub_dir, f"expert_scales.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            x_active_mask = torch.load(os.path.join(case_sub_dir, f"x_active_mask.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
            golden_expert_token_nums = torch.load(os.path.join(case_sub_dir, f"expert_token_nums.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]

            golden_x = torch.load(os.path.join(case_sub_dir, f"x_final.pt"), map_location='cpu', mmap=True, weights_only=False)[rank]
                    
            print(f'[INFO] device_{rank} 构造两算子输入数据')

            distribute_buffer = copy.copy(base_distribute_buffer)
            distribute_buffer.num_experts = moe_expert_num
            distribute_buffer.num_max_tokens_per_rank = global_bs
            distribute_buffer.num_topk = k
            distribute_buffer.hidden = h
            distribute_buffer.intermediate_hidden = h
            distribute_buffer.max_recv_token_num = max_recv_token_num
            distribute_buffer.dispatch_quant_mode = quant_mode
            distribute_buffer.dispatch_quant_out_dtype = dispatch_quant_out_type
            distribute_buffer.combine_quant_mode = comm_quant_mode
            distribute_buffer.topk_weights_type = 1 if topk_weights_type else 0
            distribute_buffer.comm_alg = ""
            cascade_kwargs = None
            mega_moe_kwargs = None
            for owner_rank in range(world_size):
                if rank == owner_rank:
                    print(f"[INFO] rank {rank} 开始分块重建并搬运权重", flush=True)
                    rebuilt_weights = rebuild_rank_weights_to_npu(ep_id)
                    weight1 = rebuilt_weights["weight1"]
                    weight2 = rebuilt_weights["weight2"]
                    weight1_scale = rebuilt_weights["weight1_scale"]
                    weight2_scale = rebuilt_weights["weight2_scale"]
                    shared_weight1 = rebuilt_weights["shared_weight1"]
                    shared_weight2 = rebuilt_weights["shared_weight2"]
                    shared_weight1_scale = rebuilt_weights["shared_weight1_scale"]
                    shared_weight2_scale = rebuilt_weights["shared_weight2_scale"]
                    cascade_kwargs = get_cascade_kwargs(
                        rank, ep_group_dispatch, tp_group_dispatch,
                        weight1, weight2, weight1_scale, weight2_scale,
                        shared_weight1, shared_weight2, shared_weight1_scale, shared_weight2_scale,
                        x, expert_ids, expert_scales, x_active_mask,
                        weights_are_prepared=True,
                    )
                    mega_moe_kwargs = get_mega_moe_kwargs(
                        rank, distribute_buffer,
                        weight1, weight2, weight1_scale, weight2_scale,
                        shared_weight1, shared_weight2, shared_weight1_scale, shared_weight2_scale,
                        expert_ids, x, expert_scales, x_active_mask,
                        weights_are_prepared=True,
                    )
                    del rebuilt_weights
                    del weight1, weight2, weight1_scale, weight2_scale
                    del shared_weight1, shared_weight2, shared_weight1_scale, shared_weight2_scale
                    gc.collect()
                    print(f"[INFO] rank {rank} 分块权重准备完成", flush=True)
                dist.barrier()
            print(f'rank:{rank} 构造算子输入finished!')
            if graph_type == 0:
                cascade_moe = get_cascade_graph_model_class()(**cascade_graph_kwargs)
                mega_moe_model = MegaMoeGraphModel(**cascade_graph_kwargs)
                if not is_performance:
                    # cascade_graph_kwargs['distribute_buffer'] = distribute_buffer
                    for _ in range(loop_acc_time):
                        x, dispatch_out = cascade_moe.forward(*cascade_kwargs)
                    synchronize_before_mega_moe()
                    for _ in range(loop_acc_time):
                        y, expert_token_nums = mega_moe_model.forward(**mega_moe_kwargs)
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
                    synchronize_before_mega_moe()
                    with get_profiling(100, 1, 0, cascade_profiling_path) as prof:
                        for i in range(100):
                            #    torch.npu.synchronize()
                            y, expert_token_nums = mega_moe_model.forward(**mega_moe_kwargs)
                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)
                            prof.step()
            else:

                loop_time = 2 if is_performance and is_graph else loop_acc_time
                if is_performance:
                    repeat_time = 1
                    warm_time = 1
                    prof = get_profiling(loop_time, repeat_time, warm_time, cascade_profiling_path)
                    prof.start()
            
                cascade_raw_model = prepare_cascade_model_for_graph(
                    get_cascade_graph_model_class()(**cascade_graph_kwargs), cascade_kwargs
                )
                cascade_moe_model = compile_model(cascade_raw_model)
                for i in range(loop_time):
                    # 调用小算子
                    x, dispatch_out = cascade_moe_model(*cascade_kwargs)

                    if is_performance:
                        prof.step()
                if is_performance:
                    prof.stop()

                synchronize_before_mega_moe()
                mega_moe_raw_model = prepare_mega_moe_model_for_graph(MegaMoeGraphModel(**cascade_graph_kwargs), mega_moe_kwargs)
                mega_moe_model = compile_model(mega_moe_raw_model)
                if is_performance:
                    repeat_time = 1
                    warm_time = 1
                    prof = get_profiling(loop_time, repeat_time, warm_time, cascade_profiling_path)
                    prof.start()
            
                for i in range(loop_time):
                    # 调用融合算子
                    y, expert_token_nums = mega_moe_model(**mega_moe_kwargs)
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
                real_tokens = x_active_mask.cpu().sum().item()
                x = x[:real_tokens, :]
                y = y[:real_tokens, :]
                golden_x = golden_x[:real_tokens, :]

            result_mega_moe = {}
            # 比较y

            res = compare_output_triangle(
                ComparedOutput.X,
                x,
                golden_x,
                y,
                torch.float32,
                True,
                rank
            )
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
            result_mega_moe.update(res)

            print(f"##################Compare x in rank,{rank}##################")

            rst_npu_x = toolss_pro.checkResult(y, golden_x, "npu_x")
            if print_compare_details:
                rst_npu_x.print_result()
            rst_npu_golden_x = toolss_pro.checkResult(x, golden_x, "npu_golden_x")
            if print_compare_details:
                rst_npu_golden_x.print_result()
            str1, str2 = rst_npu_x.check_result(rst_npu_golden_x)
            if 'error' in str1:
                res = 'FAIL'
            else:
                res = 'PASS'
            print('[INFO] tensor x output 精度结果为：{}'.format(res))
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            write_row_to_csv('mega_moe_result.csv', [current_case_name, f"output_x_rank{rank}", res, time_str, "batch"])

            print("\n##################expert_token_nums result##################")
            print("expert_token_nums: ", result_mega_moe)
            all_result = all(result_mega_moe.values())
            
        except Exception as e:
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            write_row_to_csv('moe_distribute.csv', [data_line + 1, current_case_name, f"x_{rank}", f"('FAIL', 异常)", time_str, "batch"])
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"[ERROR] Case FAILED and SKIPPED: Line {data_line + 1}, ID {id_part}")
            print(f"[ERROR] Reason: {str(e)}")
            print(f"[ERROR] Stack Trace:\n{traceback.format_exc()}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")  

        with lock:
            shared_counters[tmp].value += 1
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
    def tensor_to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def compare(cascade, golden, npu, dtype, tensor_name, diffThd=0.001, pctThd=0.001):
        np_output_npu = tensor_to_numpy(npu)
        np_output_golden = tensor_to_numpy(golden)
        np_output_cascade = tensor_to_numpy(cascade) if cascade != 'two comparison' else cascade

        max_diff_hd = 0.0
        if dtype == torch.int8:
            max_diff_hd = 1.0
        if isinstance(np_output_cascade, str) and np_output_cascade == 'two comparison':
            res_output = toolss_pro.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)
            print(f'[INFO] tensor {tensor_name} output测试的精度结果为：{res_output}')
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output, time_str, "batch"])
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
def normalize_metadata_value(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_metadata_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_metadata_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def build_golden_metadata():
    return {
        "version": 4,
        "random_seed": random_seed,
        "input_kwargs": normalize_metadata_value(input_kwargs),
    }


def load_golden_metadata(case_sub_dir):
    metadata_path = os.path.join(case_sub_dir, GOLDEN_METADATA_FILE)
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError, TypeError):
        return False


def save_golden_metadata(case_sub_dir, metadata):
    metadata_path = os.path.join(case_sub_dir, GOLDEN_METADATA_FILE)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, sort_keys=True)


def remove_persisted_weight_files(case_sub_dir):
    for file_name in PERSISTED_WEIGHT_FILES:
        file_path = os.path.join(case_sub_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def get_case_dataframe(file_path, sheet_name):
    cache_key = (os.path.abspath(file_path), sheet_name)
    if cache_key not in _xlsx_cache:
        _xlsx_cache[cache_key] = pd.read_excel(file_path, sheet_name=sheet_name).fillna('')
    return _xlsx_cache[cache_key]


def read_case_info(file_path, sheet_name, row_number, keys=None):
    try:
        xlsx_df = get_case_dataframe(file_path, sheet_name)

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
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="输出根目录")
    parser.add_argument("-g", "--golden", action="store_true", help="仅生成 golden 和输入数据")
    args = parser.parse_args()
    excel_max_row = len(get_case_dataframe(args.excel, args.sheet))
    args.end_line = min(args.end_line, excel_max_row + 1) 
    max_buff_size = 0
    if args.excel and args.sheet and args.start_line and args.end_line:  
        for data_line in range(args.start_line - 1, args.end_line):
            data = read_case_info(args.excel, args.sheet, data_line)
            init_case_info(data)
            # buff_size = get_mega_moe_ccl_buffer_size(
            #     ep_world_size, moe_expert_num, bs, k, h,
            #     # max_recv_token_num=bs * ep_world_size * min(local_moe_expert_num, k),
            #     dispatch_quant_mode=quant_mode, 
            #     dispatch_quant_out_dtype=dispatch_quant_out_type, 
            #     combine_quant_mode=comm_quant_mode, 
            #     comm_alg=comm_alg
            # )
            # max_buff_size = max(max_buff_size, buff_size)
            # 生成golden和输入
            case_sub_dir = os.path.join("./input", args.sheet, current_case_name)

            need_generate_golden = (
                not os.path.exists(case_sub_dir)
                or any(
                    not os.path.isfile(os.path.join(case_sub_dir, file_name))
                    for file_name in GOLDEN_REQUIRED_FILES
                )
            )
            expected_metadata = build_golden_metadata()
            saved_metadata = None
            if not need_generate_golden:
                saved_metadata = load_golden_metadata(case_sub_dir)
                if saved_metadata is False:
                    need_generate_golden = True
                elif saved_metadata is not None:
                    need_generate_golden = saved_metadata != expected_metadata

            if not need_generate_golden and saved_metadata is None:
                try:
                    saved_expert_token_nums = torch.load(
                        os.path.join(case_sub_dir, "expert_token_nums.pt"),
                        map_location='cpu',
                        mmap=True,
                        weights_only=False,
                    )[0]
                    topk_weights_type_path = os.path.join(case_sub_dir, "topk_weights_type.pt")
                    saved_topk_weights_type = bool(torch.load(
                        topk_weights_type_path,
                        map_location='cpu',
                        weights_only=False,
                    ))
                    need_generate_golden = (
                        saved_expert_token_nums.numel() != local_moe_expert_num
                        or saved_topk_weights_type != bool(topk_weights_type)
                    )
                    if not need_generate_golden:
                        save_golden_metadata(case_sub_dir, expected_metadata)
                except Exception:
                    need_generate_golden = True

            if os.path.isdir(case_sub_dir):
                remove_persisted_weight_files(case_sub_dir)

            if need_generate_golden:
                # os.system(f"rm -rf {case_sub_dir}")
                os.makedirs(case_sub_dir, exist_ok=True)

                # 设置随机种子
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                input_data = GenInputAndGolden(**input_kwargs)
                # 公共输入输出
                torch.save(input_data.x_final, os.path.join(case_sub_dir, f"x_final.pt"), pickle_protocol=4)
                torch.save(input_data.x, os.path.join(case_sub_dir, f"x.pt"), pickle_protocol=4)
                torch.save(input_data.expert_ids, os.path.join(case_sub_dir, f"expert_ids.pt"), pickle_protocol=4)
                torch.save(input_data.expert_scales, os.path.join(case_sub_dir, f"expert_scales.pt"), pickle_protocol=4)
                torch.save(input_data.expert_token_nums, os.path.join(case_sub_dir, f"expert_token_nums.pt"), pickle_protocol=4)
                torch.save(input_data.x_active_mask, os.path.join(case_sub_dir, f"x_active_mask.pt"), pickle_protocol=4)
                torch.save(bool(topk_weights_type), os.path.join(case_sub_dir, "topk_weights_type.pt"), pickle_protocol=4)
                save_golden_metadata(case_sub_dir, expected_metadata)
                remove_persisted_weight_files(case_sub_dir)
                del input_data
                gc.collect()
                
                print(f"成功构造第{data_line + 1}行{current_case_name}的对应golden", flush = True)

        print(f"{args.start_line} — {args.end_line}行golden已全部生成")
        # 设置 HCCL_BUFFSIZE 大小
        os.environ['HCCL_BUFFSIZE'] = str(2000)
        # print(f'cur max_buff_size: {max_buff_size}')

        if args.golden:
            print("golden and input data generated; skipping NPU execution")
            sys.exit(0)

        # npu计算及精度对比
        if args.run_mode == 0:  
            gen_npu(
                run_cascade_npu, args.excel, args.sheet, args.start_line, args.end_line
            )
        else:   # 可以单独起通信域 如需要打印性能数据
            for data_line in range(args.start_line, args.end_line + 1):
                gen_npu(
                    run_cascade_npu, args.excel, args.sheet, data_line, data_line
                )
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
