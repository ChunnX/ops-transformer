import os
# 导入torch之前设置
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "torch_extensions_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir

import csv
import hashlib
import json
import queue
import time
import unicodedata
import torch
import socket
import torch_npu
import numpy as np
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from enum import Enum
from types import SimpleNamespace
import torchair
from torchair.configs.compiler_config import CompilerConfig

import toolss_pro
from graph import CascadeGraphModel as PrefillCascadeGraphModel, MegaMoeGraphModel
from graph_decode import CascadeGraphModel as DecodeCascadeGraphModel
from gen_input_and_golden import GenInputAndGolden, mx_array_to_torch
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe


def get_cascade_graph_model_class():
    return PrefillCascadeGraphModel if moe_stage == "prefill" else DecodeCascadeGraphModel

GOLDEN_FFN_CHUNK_SIZE = 16
GOLDEN_LOG_INTERVAL = 512
GOLDEN_LOG_STATS = False
GOLDEN_BATCH_MATMUL = False
GOLDEN_BATCH_MATMUL_MIN_M = 128
GOLDEN_TORCH_NUM_INTEROP_THREADS = 1
_torch_interop_threads_configured = False


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

x_active_mask_dim = 0  # mask的维度，当前支持0,1,2

graph_type = 0  # 0为单算子，1为静态图模式，2为动态图模式, 3为aclgraph
moe_stage = "prefill"  # prefill走graph.py，decode走graph_decode.py
is_graph = (graph_type != 0)
quant_mode = 4  # 0为dispatch算子非量化场景，1为静态量化（当前不支持），2为动态量化
comm_quant_mode = 0  # 0无量化，1则combine通信采用int12动态量化（不支持），2为int8动态量化, 3为e5m2, 4为e4m3
ep_buffersize = 6000
tp_buffersize = 1000

is_performance = 0  # 是否进行性能检测
loop_acc_time = 1  # 非性能模式连续执行轮数
loop = 105  # 性能检测时的性能采集轮数

is_variable_bs = 0  # 是否使能可变bs
is_full_core = True  # True时为全核模式，使用全部的核，False时使用部分核

core_num = 18 if not is_full_core else 36  # 非全核模式使用的核数（单个核上有一个aic核，一个aiv核）

aic_core_num = core_num  # 非全核模式使用的aic核数
aiv_core_num = core_num * 2  # 非全核模式使用的aiv核数
server_num = 1  # 单机或者多机模式
tp_world_size = 1  # 1或2, 2时dispatch算子在tp域执行allgather, combine算子执行reducescatter, 1时均不执行
expert_per_rank = 4
local_moe_expert_num = 4

bs = 1

shared_expert_num = expert_per_rank - local_moe_expert_num
enable_shared_expert_multistream = True
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
moe_expert_num = ep_world_size * local_moe_expert_num  # moe专家数

h = 4096  # 每个token的长度
n = 1024
k = 6
comm_alg = ''

transpose_weight = 1

global_bs = bs * ep_world_size

out_dtype = torch.bfloat16  # 控制dispatch输入及combine输出的数据类型， 0为bfloat16，1为float16

quant_dtype_dict = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "float4_e2m1": torch.uint8,  # float4存储为uint8 (2个float4打包成1个byte)
}

swiglu_quant_dict = {
    "float8_e4m3fn": 24,
    "float8_e5m2": 23,
    "float4_e2m1": 296,
}

dispatch_quant_out_type_dict = {
    "float8_e4m3fn": 24,
    "float8_e5m2": 23,
    "float4_e2m1": 296,
}
swiglu_quant_dtype = "float8_e4m3fn"  # 当前仅支持swiglu_quant_dict表量化模式，swiglu量化模式
quant_dtype = "float8_e4m3fn"    # 当前仅支持quant_dtype_dict表量化模式，dispatch量化模式
weight_dtype = "float8_e4m3fn"    # 当前仅支持quant_dtype_dict表量化模式，weight量化模式
A8W8_is_NZ = False
clamp_limit = 9999
topk_weights_type = False
golden_cache_enable = True
golden_cache_dir = os.path.join(script_dir, "golden_cache")
golden_ffn_chunk_size = GOLDEN_FFN_CHUNK_SIZE
fast_random_weight = True
golden_batch_matmul = GOLDEN_BATCH_MATMUL
golden_batch_matmul_min_m = GOLDEN_BATCH_MATMUL_MIN_M
golden_log_interval = GOLDEN_LOG_INTERVAL
golden_log_stats = GOLDEN_LOG_STATS
golden_cpu_count, golden_expert_workers, golden_torch_num_threads = get_golden_parallel_config(golden_ffn_chunk_size)
golden_torch_num_interop_threads = GOLDEN_TORCH_NUM_INTEROP_THREADS
set_torch_cpu_threads(golden_torch_num_threads, golden_torch_num_interop_threads)
dispatch_quant_out_type = dispatch_quant_out_type_dict[quant_dtype]
is_topk_balance = True
is_x_random = True  # 是否让x值随机
random_seed = 831  # 随机种子，用于固定各随机输入变量的产生，如x,expert_ids,expert_scales等
expert_scales_dtype = torch.float32
current_case_name = 'MOE_Dis_Com_16p_bs12_k6_xdtype0_quant2_graph0_scale1_cascade0_tp1_localmoe3_share1_moe21_ID034'

ep_ranks_list = [list(range(tp_id, world_size, tp_world_size)) for tp_id in range(tp_world_size)]
tp_ranks_list = [list(range(ep_id * tp_world_size, (ep_id + 1) * tp_world_size)) for ep_id in range(ep_world_size)]
server_ranks = slice(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)

print_compare_details = False

# buff_size = get_mega_moe_ccl_buffer_size(
#     ep_world_size, moe_expert_num, bs, k, h,
#     # max_recv_token_num=bs * ep_world_size * min(local_moe_expert_num, k),
#     dispatch_quant_mode=quant_mode, 
#     dispatch_quant_out_dtype=dispatch_quant_out_type, 
#     combine_quant_mode=comm_quant_mode, 
#     comm_alg=comm_alg
# )
os.environ['HCCL_BUFFSIZE'] = f'{2000}'

print('======================================================')
print(f'{current_case_name=}')
# print(f'{buff_size=}')
print(f'{bs=}')
print(f'{h=}')
print(f'{n=}')
print(f'{k=}')
print(f'{expert_per_rank=}')
print(f'{local_moe_expert_num=}')
print(f'{shared_expert_num=}')
print(f'{enable_shared_expert_multistream=}')
print(f'{random_seed=}')
print(f'{dispatch_quant_out_type=}')
print(f'{expert_scales_dtype=}')
print(f'{quant_dtype=}')
print(f'{out_dtype=}')
print(f'{transpose_weight=}')
print(f'{is_performance=}')
print(f'{loop_acc_time=}')
print(f'{moe_stage=}')
print(f'{graph_type=}')
print(f'{quant_mode=}')
print(f'{swiglu_quant_dtype=}')
print(f'{weight_dtype=}')
print(f'{A8W8_is_NZ=}')
print(f'{clamp_limit=}')
print(f'{topk_weights_type=}')
print(f'{golden_cache_enable=}')
print(f'{golden_cache_dir=}')
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

class ComparedOutput(Enum):
    EXPERT_TOKEN_NUMS = "expert_token_nums"
    X = "x"

cascade_moe_profiling_path = ''
mega_moe_profiling_path = ''
mega_moe_op_type = 'MegaMoe'
small_ops_wall_op_type = 'small_ops_wall'
small_ops_resource_op_type = 'small_ops_resource'
communication_op_type = 'communication'
moe_expert_op_type = 'moe_expert'
shared_expert_op_type = 'shared_expert'
add_op_type = 'Add'
path_to_op_types = {}

def refresh_profiling_paths():
    global cascade_moe_profiling_path, mega_moe_profiling_path, path_to_op_types
    timestamp = os.environ.get('PROFILING_TIMESTAMP', time.strftime("%Y%m%d_%H%M%S"))
    cascade_moe_profiling_path = f'./Profiling_Result_mc2_cascade_moe_{rank_per_dev}p_{timestamp}'
    mega_moe_profiling_path = f'./Profiling_Result_mc2_mega_moe_{rank_per_dev}p_{timestamp}'
    path_to_op_types = {
        cascade_moe_profiling_path: [
            small_ops_wall_op_type, small_ops_resource_op_type,
            communication_op_type, moe_expert_op_type, shared_expert_op_type,
            add_op_type
        ],
        mega_moe_profiling_path: [mega_moe_op_type]
    }

input_kwargs = {
    'bs': bs, 'h': h, 'n': n, 'quant_dtype': quant_dtype,
    'is_x_random': is_x_random, 'global_bs': global_bs, 'out_dtype': out_dtype, 'moe_expert_num': moe_expert_num,
    'is_topk_balance': is_topk_balance, 'k': k, 'is_mask': is_mask, 'is_unfixed_real_token': is_unfixed_real_token,
    'ep_world_size': ep_world_size, 'is_variable_bs': is_variable_bs, 'expert_scales_dtype': expert_scales_dtype,
    'local_moe_expert_num': local_moe_expert_num, 'expert_per_rank': expert_per_rank,
    'shared_expert_num': shared_expert_num, 'quant_mode': quant_mode, 'transpose_weight': transpose_weight,
    'comm_quant_mode': comm_quant_mode, 'weight_dtype': weight_dtype, 'swiglu_quant_dtype': swiglu_quant_dtype,
    'clamp_limit': clamp_limit, 'golden_ffn_chunk_size': golden_ffn_chunk_size, 'fast_random_weight': fast_random_weight,
    'golden_batch_matmul': golden_batch_matmul, 'golden_batch_matmul_min_m': golden_batch_matmul_min_m,
    'golden_expert_workers': golden_expert_workers, 'golden_log_interval': golden_log_interval,
    'golden_log_stats': golden_log_stats, 'topk_weights_type': topk_weights_type,
    'random_seed': random_seed,
}

cascade_graph_kwargs = {
    'loop': loop, 'is_performance': is_performance, "aic_core_num": aic_core_num, "aiv_core_num": aiv_core_num,
    "is_graph": is_graph, "graph_type": graph_type, "clamp_limit": clamp_limit,
    "local_moe_expert_num": local_moe_expert_num, "shared_expert_num": shared_expert_num,
    "expert_per_rank": expert_per_rank, "enable_shared_expert_multistream": enable_shared_expert_multistream,
    "topk_weights_type": topk_weights_type,
}

def write_row_to_csv(csv_name: str, row_content: list):
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)

def normalize_cache_value(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalize_cache_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_cache_value(v) for v in value]
    if isinstance(value, set):
        return sorted(normalize_cache_value(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)

def get_file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def summarize_cached_value(value):
    if isinstance(value, torch.Tensor):
        return {"type": "torch.Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"type": "np.ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, list):
        return [summarize_cached_value(v) for v in value]
    if isinstance(value, tuple):
        return [summarize_cached_value(v) for v in value]
    if value is None:
        return None
    return {"type": type(value).__name__}

def summarize_input_data(input_data):
    summary_names = [
        "x", "expert_ids", "expert_scales", "x_active_mask", "expert_token_nums",
        "w1_npu", "w2_npu", "weight_scales1", "weight_scales2",
        "shared_weight1", "shared_weight2", "shared_weight_scales1", "shared_weight_scales2",
        "x_final",
    ]
    return {name: summarize_cached_value(getattr(input_data, name, None)) for name in summary_names}

def build_golden_cache_payload():
    gen_input_path = os.path.join(script_dir, "gen_input_and_golden.py")
    return {
        "cache_version": 1,
        "random_seed": random_seed,
        "input_kwargs": normalize_cache_value(input_kwargs),
        "gen_input_and_golden_sha256": get_file_sha256(gen_input_path),
    }

def get_golden_cache_paths(cache_key):
    cache_case_dir = os.path.join(golden_cache_dir, cache_key)
    return (
        cache_case_dir,
        os.path.join(cache_case_dir, "metadata.json"),
        os.path.join(cache_case_dir, "input_data.pt"),
    )

def golden_cache_payload_matches(saved_payload, cache_payload):
    if not isinstance(saved_payload, dict):
        return False
    return (
        saved_payload.get("cache_version") == cache_payload.get("cache_version")
        and saved_payload.get("random_seed") == cache_payload.get("random_seed")
        and saved_payload.get("input_kwargs") == cache_payload.get("input_kwargs")
        and saved_payload.get("gen_input_and_golden_sha256") == cache_payload.get("gen_input_and_golden_sha256")
    )

def find_matching_golden_cache(cache_payload):
    if not os.path.isdir(golden_cache_dir):
        return None
    for cache_name in sorted(os.listdir(golden_cache_dir)):
        cache_case_dir = os.path.join(golden_cache_dir, cache_name)
        metadata_path = os.path.join(cache_case_dir, "metadata.json")
        data_path = os.path.join(cache_case_dir, "input_data.pt")
        if not os.path.isfile(metadata_path) or not os.path.isfile(data_path):
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            continue
        if golden_cache_payload_matches(metadata.get("cache_payload"), cache_payload):
            return cache_case_dir, metadata_path, data_path
    return None

def load_cached_input_data(data_path):
    try:
        return torch.load(data_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(data_path, map_location="cpu")

def save_cached_input_data(input_data, metadata_path, data_path, metadata):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    tmp_data_path = data_path + ".tmp"
    tmp_metadata_path = metadata_path + ".tmp"
    torch.save(input_data, tmp_data_path, pickle_protocol=4)
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_data_path, data_path)
    os.replace(tmp_metadata_path, metadata_path)

def load_or_generate_input_data():
    cache_payload = build_golden_cache_payload()
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_case_dir, metadata_path, data_path = get_golden_cache_paths(cache_key)

    if golden_cache_enable and os.path.exists(metadata_path) and os.path.exists(data_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if golden_cache_payload_matches(metadata.get("cache_payload"), cache_payload):
                print(f"[INFO] Golden cache hit: {cache_case_dir}")
                return load_cached_input_data(data_path)
            print(f"[INFO] Golden cache metadata mismatch, regenerate: {cache_case_dir}")
        except Exception as exc:
            print(f"[WARN] Golden cache load failed, regenerate: {exc}")

    if golden_cache_enable:
        matched_cache = find_matching_golden_cache(cache_payload)
        if matched_cache is not None:
            cache_case_dir, _, data_path = matched_cache
            print(f"[INFO] Golden cache hit by same input variables: {cache_case_dir}")
            return load_cached_input_data(data_path)

    print("[INFO] Golden cache miss, generate input and golden.")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    golden_start_time = time.perf_counter()
    input_data = GenInputAndGolden(**input_kwargs)
    golden_elapsed_time = time.perf_counter() - golden_start_time
    print(f"[INFO] Golden generate time: {golden_elapsed_time:.3f}s")

    if golden_cache_enable:
        metadata = {
            "cache_key": cache_key,
            "cache_payload": cache_payload,
            "case_name": current_case_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "golden_generate_seconds": golden_elapsed_time,
            "input_summary": summarize_input_data(input_data),
        }
        save_cached_input_data(input_data, metadata_path, data_path, metadata)
        print(f"[INFO] Golden cache saved: {cache_case_dir}")

    return input_data

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

    if loop_acc_time < 1:
        raise ValueError(f"loop_acc_time must be >= 1, got {loop_acc_time}")

    if graph_type == 0 and is_performance:
        print('单算子模式(graph_type=0)不支持性能采集，请使用图模式进行性能采集!')
        exit(0)

    if k > moe_expert_num:
        print('k大于总的moe专家数!')
        exit(0)

    if moe_expert_num % ep_world_size != 0:
        print(f"moe_expert_num({moe_expert_num}) must be divisible by ep_world_size({ep_world_size})")
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
    return [
        min(per_rank_medians),
        max(per_rank_medians),
        float(np.median(per_rank_medians)),
        min(per_rank_mins),
    ]


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
            # Pre-quantized dispatch uses two routing calls. Only the first one
            # has a dispatch quant/all-gather prefix and starts a new iteration.
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
            "start_us": float(segment["start_us"].min()),
            "end_us": float(segment["end_us"].max()),
            "wall_span_us": wall_span_us,
            "resource_sum_us": resource_sum_us,
            "communication_us": float(duration_by_group.get(communication_op_type, 0.0)),
            "moe_expert_us": float(duration_by_group.get(moe_expert_op_type, 0.0)),
            "shared_expert_us": float(duration_by_group.get(shared_expert_op_type, 0.0)),
            "add_us": float(duration_by_group.get(add_op_type, 0.0)),
            "overlap_saved_us": max(0.0, resource_sum_us - wall_span_us),
            "op_count": int(len(segment)),
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
    return performance_info

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

def print_cascade_performance(op_types, op_performances):
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
    compare_rows = [
        ['小算子真实耗时(wall)', format_perf_value(small_wall[2])],
        ['小算子耗时(sum)', format_perf_value(small_resource[2])],
        ['融合MegaMoe', format_perf_value(mega_moe[2])],
    ]
    print_perf_table('[INFO] 性能汇总(us)', ['指标', '中位数的中位数'], compare_rows)


def get_cascade_performance():
    cascade_op_types = path_to_op_types[cascade_moe_profiling_path]
    mega_moe_op_types = path_to_op_types[mega_moe_profiling_path]
    cascade_moe_performance = get_performance(cascade_moe_profiling_path)
    mega_moe_performance = get_performance(mega_moe_profiling_path)

    op_types = cascade_op_types + mega_moe_op_types
    op_performances = cascade_moe_performance + mega_moe_performance
    print_cascade_performance(op_types, op_performances)
    save_performance_result('cascade', current_case_name, op_types, op_performances)

def save_performance_result(test_mode, case_name, op_types, op_performances):
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
            if not os.path.exists(f"csv_result/{csv_name}"):
                write_row_to_csv(csv_name, title)
            write_row_to_csv(csv_name, row)

    save_execution_time_result(test_mode, case_name, op_types, op_performances)

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

def build_cascade_expert_inputs(expert_ids, expert_scales):
    return expert_ids.contiguous(), expert_scales.contiguous(), moe_expert_num

def get_rank_attr(input_data, name, rank):
    value = getattr(input_data, name)
    if isinstance(value, (list, tuple)):
        return value[rank]
    return value

def clone_rank_attr(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value

def make_rank_input_data(input_data, rank):
    rank_fields = (
        "x", "expert_ids", "expert_scales", "x_active_mask",
        "w1_npu", "w2_npu", "weight_scales1", "weight_scales2",
        "shared_weight1", "shared_weight2", "shared_weight_scales1", "shared_weight_scales2",
    )
    return SimpleNamespace(**{
        field: clone_rank_attr(get_rank_attr(input_data, field, rank))
        for field in rank_fields
    })

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

def get_cascade_kwargs(rank, input_data, group_ep, group_tp):
    ep_rank_id = rank // tp_world_size
    w1_npu = get_rank_attr(input_data, "w1_npu", rank)
    w2_npu = get_rank_attr(input_data, "w2_npu", rank)
    weight_scales1 = get_rank_attr(input_data, "weight_scales1", rank)
    weight_scales2 = get_rank_attr(input_data, "weight_scales2", rank)
    shared_weight1 = get_rank_attr(input_data, "shared_weight1", rank)
    shared_weight2 = get_rank_attr(input_data, "shared_weight2", rank)
    shared_weight_scales1 = get_rank_attr(input_data, "shared_weight_scales1", rank)
    shared_weight_scales2 = get_rank_attr(input_data, "shared_weight_scales2", rank)
    weight1 = combine_cascade_weight_inputs(w1_npu, shared_weight1, quant_dtype)
    weight2 = combine_cascade_weight_inputs(w2_npu, shared_weight2, quant_dtype)
    weight1_scale = combine_cascade_weight_inputs(
        weight_scales1, shared_weight_scales1, quant_dtype, is_scale=True
    )
    weight2_scale = combine_cascade_weight_inputs(
        weight_scales2, shared_weight_scales2, quant_dtype, is_scale=True
    )
    weight1_for_gmm = prepare_cascade_weight_partitions(weight1, quant_dtype)
    weight2_for_gmm = prepare_cascade_weight_partitions(weight2, swiglu_quant_dtype)
    cascade_expert_ids, cascade_expert_scales, cascade_expert_num = build_cascade_expert_inputs(
        get_rank_attr(input_data, "expert_ids", rank),
        get_rank_attr(input_data, "expert_scales", rank),
    )
    return [
        get_rank_attr(input_data, "x", rank).npu(), cascade_expert_ids.npu(),
        cascade_expert_scales.npu(),
        get_rank_attr(input_data, "x_active_mask", rank).npu(), group_ep, group_tp, ep_rank_id, ep_world_size, cascade_expert_num, quant_mode,
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
        "max_recv_token_num": bs * ep_world_size * min(local_moe_expert_num, k),
        "dispatch_quant_mode": quant_mode,
        "dispatch_quant_out_dtype": dispatch_quant_out_type,
        "combine_quant_mode": comm_quant_mode,
        # "topk_weights_type": 1 if topk_weights_type else 0,
        "comm_alg": ""
        }

def get_mega_moe_kwargs(rank, input_data, distribute_buffer):
    w1_npu = get_rank_attr(input_data, "w1_npu", rank)
    w2_npu = get_rank_attr(input_data, "w2_npu", rank)
    weight_scales1 = get_rank_attr(input_data, "weight_scales1", rank)
    weight_scales2 = get_rank_attr(input_data, "weight_scales2", rank)
    weight1_for_gmm = prepare_moe_weight(restore_weight_input(w1_npu, quant_dtype), quant_dtype)
    weight2_for_gmm = prepare_moe_weight(restore_weight_input(w2_npu, quant_dtype), swiglu_quant_dtype)
    weight1_scale = restore_weight_scale_input(weight_scales1)
    weight2_scale = restore_weight_scale_input(weight_scales2)
    mega_moe_kwargs = {
        "x": get_rank_attr(input_data, "x", rank).npu(),
        "topk_ids": get_rank_attr(input_data, "expert_ids", rank).npu(),
        "topk_weights": get_rank_attr(input_data, "expert_scales", rank).npu(),
        "l1_weights": weight1_for_gmm,
        "l2_weights": weight2_for_gmm,
        "sym_buffer": distribute_buffer,
        "l1_weights_sf": None if quant_mode==0 else weight1_scale.npu(),
        "l2_weights_sf": None if quant_mode==0 else weight2_scale.npu(),
        "x_active_mask": get_rank_attr(input_data, "x_active_mask", rank).npu(),
        "weight1_type": dispatch_quant_out_type_dict[weight_dtype],
        "weight2_type": dispatch_quant_out_type_dict[weight_dtype],
        "rank": rank,
    }
    if shared_expert_num > 0:
        shared_weight1 = prepare_moe_weight(
            restore_weight_input(get_rank_attr(input_data, "shared_weight1", rank), quant_dtype),
            quant_dtype,
        )
        shared_weight2 = prepare_moe_weight(
            restore_weight_input(get_rank_attr(input_data, "shared_weight2", rank), quant_dtype),
            swiglu_quant_dtype,
        )
        shared_weight_scales1 = restore_weight_scale_input(
            get_rank_attr(input_data, "shared_weight_scales1", rank)
        )
        shared_weight_scales2 = restore_weight_scale_input(
            get_rank_attr(input_data, "shared_weight_scales2", rank)
        )
        mega_moe_kwargs.update({
            "shared_weight1": shared_weight1,
            "shared_weight2": shared_weight2,
            "shared_weight_scales1": None if quant_mode == 0 else shared_weight_scales1.npu(),
            "shared_weight_scales2": None if quant_mode == 0 else shared_weight_scales2.npu(),
        })
    return mega_moe_kwargs
    

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


def extract_kernel_timing(output_tensor, num_cores, rank, out_dir=None, cycles_to_us=50):
    if out_dir is None:
        out_dir = "."
    out_dir = os.path.join(out_dir, f"rank{rank}")
    os.makedirs(out_dir, exist_ok=True)

    timing_slots = 2
    timing_bytes = num_cores * timing_slots * 2 * 8
    timing = output_tensor.contiguous().view(torch.uint8).reshape(-1)[:timing_bytes].cpu().view(torch.int64).reshape(num_cores, timing_slots, 2)

    raw_columns = []
    for i in range(1, timing_slots + 1):
        raw_columns.extend([f"slot_{i}_start", f"slot_{i}_stop"])
    index = [f"core_{c}" for c in range(num_cores)]
    raw_data = timing.numpy().reshape(num_cores, -1).astype(np.int64)
    df_raw = pd.DataFrame(raw_data, columns=raw_columns, index=index)
    raw_csv_path = os.path.join(out_dir, f"timing_rank_{rank}_raw.csv")
    df_raw.to_csv(raw_csv_path)
    print(f"[TIMING] rank {rank}: raw table saved to {raw_csv_path}")

    min_slot1_start = raw_data[:, 0].min()
    relative_data = np.where(raw_data != 0, raw_data - min_slot1_start, 0)
    df_rel = pd.DataFrame(relative_data, columns=raw_columns, index=index)
    rel_csv_path = os.path.join(out_dir, f"timing_rank_{rank}_relative.csv")
    df_rel.to_csv(rel_csv_path)
    print(f"[TIMING] rank {rank}: relative table saved to {rel_csv_path}")

    duration = timing.numpy()[:, :, 1] - timing.numpy()[:, :, 0]
    duration_us = np.round(duration / cycles_to_us, 2)
    dur_columns = [f"slot_{i}" for i in range(1, timing_slots + 1)]
    df_dur = pd.DataFrame(duration_us, columns=dur_columns, index=index)
    dur_csv_path = os.path.join(out_dir, f"timing_rank_{rank}_duration.csv")
    df_dur.to_csv(dur_csv_path)
    print(f"[TIMING] rank {rank}: duration table saved to {dur_csv_path}")

    slot_names = ["blockMmad", "swigluQuantOp"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[TIMING] rank {rank}: ERROR — missing Python module: {e.name}. "
              f"Please install it with: pip install {e.name}")
        return timing

    try:
        data = timing.numpy().astype(np.float64)
        duration_us = (data[:, :, 1] - data[:, :, 0]) / cycles_to_us

        active_slots = np.any(duration_us > 0, axis=0)
        n_active = active_slots.sum()
        if n_active > 0:
            slot_ids = np.where(active_slots)[0]

            cmap = plt.cm.tab20
            colors = [cmap(i % 20) for i in range(len(slot_ids))]

            fig2, ax2 = plt.subplots(figsize=(14, max(5, num_cores * 0.35)))
            left = np.zeros(num_cores)
            for i, sid in enumerate(slot_ids):
                ax2.barh(range(num_cores), duration_us[:, sid], left=left,
                         height=0.7, color=colors[i], edgecolor="white", linewidth=0.3,
                         label=f"slot_{sid+1} ({slot_names[sid]})")
                left += duration_us[:, sid]

            max_total = left.max()
            min_width = max_total * 0.01 if max_total > 0 else 0
            left = np.zeros(num_cores)
            for sid in slot_ids:
                for core in range(num_cores):
                    val = duration_us[core, sid]
                    if val > min_width:
                        cx = left[core] + val / 2.0
                        ax2.text(cx, core, f"{val:.2f}", ha="center", va="center",
                                 fontsize=5, color="black")
                left += duration_us[:, sid]

            ax2.set_yticks(range(num_cores))
            ax2.set_yticklabels([f"core_{c}" for c in range(num_cores)], fontsize=7)
            ax2.set_xlabel("accumulated time (us)")
            ax2.set_title(f"Per-Core Accumulated Timing — Rank {rank}  (us)")
            ax2.invert_yaxis()
            ax2.legend(fontsize=6, ncol=2, loc="lower right")
            fig2.tight_layout()

            stack_path = os.path.join(out_dir, f"timing_rank_{rank}_stack.png")
            fig2.savefig(stack_path, dpi=150)
            plt.close(fig2)
            print(f"[TIMING] rank {rank}: stacked bar chart saved to {stack_path}")

            fig3, ax3 = plt.subplots(figsize=(16, max(5, num_cores * 0.35)))
            for i, sid in enumerate(slot_ids):
                ax3.barh(range(num_cores), duration_us[:, sid], left=np.zeros(num_cores),
                         height=0.7, color=colors[i], edgecolor="white", linewidth=0.3,
                         label=f"slot_{sid+1} ({slot_names[sid]})")

            for sid in slot_ids:
                for core in range(num_cores):
                    val = duration_us[core, sid]
                    if val > min_width:
                        ax3.text(val / 2.0, core, f"{val:.2f}", ha="center", va="center",
                                 fontsize=5, color="black")

            ax3.set_yticks(range(num_cores))
            ax3.set_yticklabels([f"core_{c}" for c in range(num_cores)], fontsize=7)
            ax3.set_xlabel("accumulated time (us)")
            ax3.set_title(f"Per-Core Accumulated Timing Breakdown — Rank {rank}  (us)")
            ax3.invert_yaxis()
            ax3.legend(fontsize=6, ncol=2, loc="lower right")
            fig3.tight_layout()

            timeline_path = os.path.join(out_dir, f"timing_rank_{rank}_timeline.png")
            fig3.savefig(timeline_path, dpi=150)
            plt.close(fig3)
            print(f"[TIMING] rank {rank}: timeline chart saved to {timeline_path}")
    except Exception as e:
        print(f"[TIMING] rank {rank}: chart error — {e}")

    return timing


def run_cascade_npu(queue, rank, input_data):
    refresh_profiling_paths()
    set_device(rank)
    _, _, _, ep_group_dispatch, tp_group_dispatch, ep_group_mega = init_hccl_comm_two_ep(rank)


    cascade_kwargs = get_cascade_kwargs(rank, input_data, ep_group_dispatch, tp_group_dispatch)
    distribute_buffer = get_symm_buffer_for_mega_moe(**get_symm_buffer_kwargs(ep_group_mega))
    mega_moe_kwargs = get_mega_moe_kwargs(rank, input_data, distribute_buffer)
    print(f'rank:{rank} 构造算子输入finished!')
    if graph_type == 0:
        cascade_moe = get_cascade_graph_model_class()(**cascade_graph_kwargs)
        mega_moe_model = MegaMoeGraphModel(**cascade_graph_kwargs)
        if not is_performance:
            for _ in range(loop_acc_time):
                x, dispatch_out = cascade_moe.forward(*cascade_kwargs)
            synchronize_before_mega_moe()
            for _ in range(loop_acc_time):
                y, expert_token_nums = mega_moe_model.forward(**mega_moe_kwargs)
        else:
            
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(100, 1, 0, cascade_moe_profiling_path) as prof:
                for i in range(100):
                    x, dispatch_out = cascade_moe.forward(*cascade_kwargs)
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()
                torch.npu.synchronize()
            synchronize_before_mega_moe()
            with get_profiling(100, 1, 0, mega_moe_profiling_path) as prof:
               for i in range(100):
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
            prof = get_profiling(loop_time, repeat_time, warm_time, cascade_moe_profiling_path)
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
            prof = get_profiling(loop_time, repeat_time, warm_time, mega_moe_profiling_path)
            prof.start()
    
        for i in range(loop_time):
            # 调用融合算子
            y, expert_token_nums = mega_moe_model(**mega_moe_kwargs)
            if is_performance:
                prof.step()
        if is_performance:
            prof.stop()
    torch.npu.synchronize()

    timing_y = y.clone()
    extract_kernel_timing(timing_y, num_cores=aic_core_num + aiv_core_num, rank=rank, out_dir="timing_output", cycles_to_us=50)

    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    # 获得x中的有效token
    if is_mask:
        real_tokens = get_rank_attr(input_data, "x_active_mask", rank).cpu().sum().item()
        x = x[:real_tokens, :]
        y = y[:real_tokens, :]
    queue.put([
        rank,
        [x.cpu(), y.cpu(), dispatch_out[3].cpu(), expert_token_nums.cpu()]
    ])

def compare_output_triangle(compared_output: ComparedOutput, cascade_per_card, golden_per_card, npu_per_card, compare_dtype, compare_per_rank):
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
            write_row_to_csv('moe_distribute.csv', [current_case_name, tensor_name, res_output])
            return res_output[0] == 'PASS'

        else:
            toolss_pro.data_compare_triangle(
                np_output_cascade, np_output_golden, np_output_npu,
                diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd
            )
            return None

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
    os.environ['PROFILING_TIMESTAMP'] = time.strftime("%Y%m%d_%H%M%S")
    refresh_profiling_paths()
    check_input()
    input_data = load_or_generate_input_data()

    if int(server_num) > 1:
        print("multi_server scene!!!!! only support running case one by one!!!!!!!!!!!!!")
        rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    else:
        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
    
    print(f"rank list is: {rank_list}")
    proc_list = []
    mp.set_start_method("spawn", force=True)
    queue_manager = mp.Manager()
    result_queue = queue_manager.Queue()
    for rank in rank_list:
        rank_input_data = make_rank_input_data(input_data, rank)
        proc = Process(target=run_cascade_npu, kwargs={"queue": result_queue, "rank": rank, "input_data": rank_input_data})
        proc.start()
        proc_list.append(proc)

    rank_outputs = [None] * rank_per_dev
    pending_results = len(proc_list)
    while pending_results:
        try:
            rank_id, rank_output = result_queue.get(timeout=1)
        except queue.Empty:
            failed_procs = [
                proc for proc in proc_list
                if proc.exitcode not in (None, 0)
            ]
            if failed_procs:
                failed_info = ", ".join(
                    f"pid={proc.pid}, exitcode={proc.exitcode}" for proc in failed_procs
                )
                raise RuntimeError(f"rank process failed before sending result: {failed_info}")
            continue
        local_rank_id = rank_id - server_index * rank_per_dev
        rank_outputs[local_rank_id] = rank_output
        pending_results -= 1

    for proc in proc_list:
        proc.join()
    queue_manager.shutdown()

    # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
        category_outputs.append(specific_category_output)

    result_mega_moe = {}
    # 比较y
    golden_x = input_data.x_final
    golden_expert_token_nums = input_data.expert_token_nums

    compare_output_triangle(
        ComparedOutput.X,
        category_outputs[0],
        golden_x[server_ranks],
        category_outputs[1],
        torch.float32,
        True,
    )

    # 比较expert_token_num
    res = compare_output_triangle(
        ComparedOutput.EXPERT_TOKEN_NUMS,
        'two comparison',
        golden_expert_token_nums[server_ranks],
        category_outputs[3],
        torch.float32,
        False,
    )
    result_mega_moe.update(res)
    for i in range(rank_per_dev):
        print("##################Compare x in rank", i, "epId", i // tp_world_size, "tpId", i % tp_world_size, "##################")

        rst_npu_x = toolss_pro.checkResult(category_outputs[1][i], golden_x[server_ranks][i], "npu_x")
        if print_compare_details:
            rst_npu_x.print_result()
        rst_npu_golden_x = toolss_pro.checkResult(category_outputs[0][i], golden_x[server_ranks][i], "npu_golden_x")
        if print_compare_details:
            rst_npu_golden_x.print_result()
        str1, _ = rst_npu_x.check_result(rst_npu_golden_x)
        if 'error' in str1:
            res = 'FAIL'
        else:
            res = 'PASS'
        print('[INFO] tensor x output 精度结果为：{}'.format(res))
        write_row_to_csv('moe_combineARN_triCompare.csv', [current_case_name, f"output_x_rank{i}", res])

    print("\n##################expert_token_nums result##################")
    print("expert_token_nums: ", result_mega_moe)
    if is_performance:
        get_cascade_performance()
