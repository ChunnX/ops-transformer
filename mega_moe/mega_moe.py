import os
# 导入torch之前设置
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "torch_extensions_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir

import csv
import gc
import hashlib
import json
import math
import time
import traceback
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
from gen_input_and_golden import GenInputAndGolden, iter_rank_npu_weight_chunks, mx_array_to_torch
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe
from mega_moe_config import CONFIG


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

# 用户可编辑参数集中在 mega_moe_config.py；这里保留兼容变量，避免执行逻辑依赖散落的配置读取。
x_active_mask_dim = CONFIG.shape.x_active_mask_dim  # mask的维度，当前支持0,1,2

graph_type = CONFIG.model.graph_type  # 0为单算子，1为静态图模式，2为动态图模式, 3为aclgraph
is_graph = (graph_type != 0)
quant_mode = CONFIG.quantization.quant_mode
comm_quant_mode = CONFIG.quantization.combine_quant_mode
ep_buffersize = CONFIG.parallel.ep_buffer_size
tp_buffersize = CONFIG.parallel.tp_buffer_size

is_performance = CONFIG.run.is_performance
loop_acc_time = CONFIG.run.functional_loop_count
loop = CONFIG.run.performance_loop_count

is_variable_bs = CONFIG.shape.is_variable_bs
is_full_core = CONFIG.parallel.is_full_core

core_num = 36 if is_full_core else CONFIG.parallel.partial_core_num

aic_core_num = core_num  # 非全核模式使用的aic核数
aiv_core_num = core_num * 2  # 非全核模式使用的aiv核数
kernel_timing_enable = CONFIG.diagnostics.kernel_timing_enable
kernel_timing_cycles_to_us = CONFIG.diagnostics.kernel_timing_cycles_to_us
kernel_timing_output_dir = os.path.join(script_dir, CONFIG.diagnostics.kernel_timing_output_dir)
critical_path_trace_enable = CONFIG.diagnostics.critical_path_trace_enable
critical_path_trace_output_dir = os.path.join(script_dir, CONFIG.diagnostics.critical_path_trace_output_dir)
server_num = CONFIG.parallel.server_num
tp_world_size = CONFIG.parallel.tp_world_size

local_moe_expert_num = CONFIG.parallel.local_moe_expert_num
shared_expert_num = CONFIG.parallel.shared_expert_num
expert_per_rank = shared_expert_num + local_moe_expert_num

enable_shared_expert_multistream = CONFIG.parallel.enable_shared_expert_multistream
is_cloud = CONFIG.parallel.is_cloud
if not is_cloud:
    server_index = CONFIG.parallel.server_index
    master_ip = CONFIG.parallel.master_ip
else:
    # 云上环境用下面方式获取server_index和master_ip
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

rank_per_dev = CONFIG.parallel.rank_per_dev
is_mask = 1 if x_active_mask_dim == 1 else 0
is_unfixed_real_token = is_mask or is_variable_bs  # 每张卡上真实token数是否固定
world_size = server_num * rank_per_dev  # 通信域总die数
ep_world_size = world_size // tp_world_size
moe_expert_num = ep_world_size * local_moe_expert_num  # moe专家数

bs = CONFIG.shape.bs
h = CONFIG.shape.h
n = CONFIG.shape.n
k = CONFIG.shape.top_k
# num_max_tokens_per_rank = bs if is_variable_bs else 0
moe_stage = "prefill" if bs > 512 or k > 16 else "decode"
comm_alg = CONFIG.model.comm_alg

transpose_weight = CONFIG.model.transpose_weight
# Enable only for cases whose fused MegaMoe kernel expects the alternating
# 128-column gate/up layout. Non-interleaved remains the compatibility default.
is_interleaved = CONFIG.model.is_interleaved

global_bs = bs * ep_world_size
performance_loop_outside_graph = global_bs > 12 * 1024

output_dtype_dict = {"bfloat16": torch.bfloat16, "float16": torch.float16}
out_dtype = output_dtype_dict[CONFIG.quantization.output_dtype]

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
swiglu_quant_dtype = CONFIG.quantization.swiglu_quant_dtype
quant_dtype = CONFIG.quantization.dispatch_quant_dtype
weight_dtype = CONFIG.quantization.weight_dtype
A8W8_is_NZ = CONFIG.quantization.a8w8_is_nz
weight_is_tensor_list = CONFIG.model.weight_is_tensor_list
clamp_limit = CONFIG.model.clamp_limit
activation_type = CONFIG.model.activation_type
swiglu_alpha = CONFIG.model.swiglu_alpha
swiglu_beta = CONFIG.model.swiglu_beta
activation_direct_output = CONFIG.model.activation_direct_output
activation_debug_max_print = CONFIG.model.activation_debug_max_print
situ_beta = CONFIG.model.situ_beta
situ_linear_beta = CONFIG.model.situ_linear_beta
is_activation_params = CONFIG.model.is_activation_params
topk_weights_type = CONFIG.quantization.topk_weights_type
golden_cache_enable = CONFIG.run.golden_cache_enable
golden_cache_dir = os.path.join(script_dir, CONFIG.run.golden_cache_dir)
golden_ffn_chunk_size = CONFIG.run.golden_ffn_chunk_size
fast_random_weight = CONFIG.run.fast_random_weight
golden_batch_matmul = CONFIG.run.golden_batch_matmul
golden_batch_matmul_min_m = CONFIG.run.golden_batch_matmul_min_m
golden_log_interval = CONFIG.run.golden_log_interval
golden_log_stats = CONFIG.run.golden_log_stats
golden_cpu_count, golden_expert_workers, golden_torch_num_threads = get_golden_parallel_config(golden_ffn_chunk_size)
golden_torch_num_interop_threads = CONFIG.run.golden_num_interop_threads
set_torch_cpu_threads(golden_torch_num_threads, golden_torch_num_interop_threads)
dispatch_quant_out_type = dispatch_quant_out_type_dict[quant_dtype]
is_topk_balance = CONFIG.run.is_topk_balance
is_x_random = CONFIG.run.is_x_random
random_seed = CONFIG.run.random_seed
expert_scales_dtype_dict = {"float32": torch.float32, "bfloat16": torch.bfloat16}
expert_scales_dtype = expert_scales_dtype_dict[CONFIG.run.expert_scales_dtype]
current_case_name = CONFIG.run.case_name

ep_ranks_list = [list(range(tp_id, world_size, tp_world_size)) for tp_id in range(tp_world_size)]
tp_ranks_list = [list(range(ep_id * tp_world_size, (ep_id + 1) * tp_world_size)) for ep_id in range(ep_world_size)]
server_ranks = slice(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)

print_compare_details = CONFIG.run.print_compare_details

# buff_size = get_mega_moe_ccl_buffer_size(
#     ep_world_size, moe_expert_num, bs, k, h,
#     # max_recv_token_num=bs * ep_world_size * min(local_moe_expert_num, k),
#     dispatch_quant_mode=quant_mode, 
#     dispatch_quant_out_dtype=dispatch_quant_out_type, 
#     combine_quant_mode=comm_quant_mode, 
#     comm_alg=comm_alg
# )
# os.environ['HCCL_BUFFSIZE'] = f'{2000}'

print('======================================================')
print(f'{current_case_name=}')
# print(f'{buff_size=}')
print(f'{bs=}')
# print(f'{num_max_tokens_per_rank=}')
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
print(f'{is_interleaved=}')
print(f'{is_performance=}')
print(f'{loop_acc_time=}')
print(f'{performance_loop_outside_graph=}')
print(f'{kernel_timing_enable=}')
print(f'{kernel_timing_cycles_to_us=}')
print(f'{kernel_timing_output_dir=}')
print(f'{moe_stage=}')
print(f'{graph_type=}')
print(f'{quant_mode=}')
print(f'{swiglu_quant_dtype=}')
print(f'{weight_dtype=}')
print(f'{A8W8_is_NZ=}')
print(f'{weight_is_tensor_list=}')
print(f'{clamp_limit=}')
print(f'{activation_type=}')
print(f'{swiglu_alpha=}')
print(f'{swiglu_beta=}')
print(f'{activation_direct_output=}')
print(f'{activation_debug_max_print=}')
print(f'{situ_beta=}')
print(f'{situ_linear_beta=}')
print(f'{is_activation_params=}')
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
compute_op_type = 'compute'
moe_expert_op_type = 'moe_expert'
shared_expert_op_type = 'shared_expert'
add_op_type = 'Add'
communication_mask_time_op_type = 'communication_mask_time'
mask_rate_op_type = 'mask_rate'
communication_mask_rate_op_type = 'communication_mask_rate'
compute_mask_rate_op_type = 'compute_mask_rate'
path_to_op_types = {}

def refresh_profiling_paths():
    global cascade_moe_profiling_path, mega_moe_profiling_path, path_to_op_types
    timestamp = os.environ.get('PROFILING_TIMESTAMP', time.strftime("%Y%m%d_%H%M%S"))
    cascade_moe_profiling_path = f'./Profiling_Result_mc2_cascade_moe_{rank_per_dev}p_{timestamp}'
    mega_moe_profiling_path = f'./Profiling_Result_mc2_mega_moe_{rank_per_dev}p_{timestamp}'
    cascade_op_types = [
        small_ops_wall_op_type, small_ops_resource_op_type,
        communication_op_type, compute_op_type, moe_expert_op_type,
    ]
    if shared_expert_num > 0:
        cascade_op_types.extend([shared_expert_op_type, add_op_type])
    path_to_op_types = {
        cascade_moe_profiling_path: cascade_op_types,
        mega_moe_profiling_path: [mega_moe_op_type]
    }

input_kwargs = {
    'bs': bs, 'h': h, 'n': n, 'quant_dtype': quant_dtype,
    'is_x_random': is_x_random, 'global_bs': global_bs, 'out_dtype': out_dtype, 'moe_expert_num': moe_expert_num,
    'is_topk_balance': is_topk_balance, 'k': k, 'is_mask': is_mask, 'is_unfixed_real_token': is_unfixed_real_token,
    'ep_world_size': ep_world_size, 'is_variable_bs': is_variable_bs, 'expert_scales_dtype': expert_scales_dtype,
    'local_moe_expert_num': local_moe_expert_num, 'expert_per_rank': expert_per_rank,
    'shared_expert_num': shared_expert_num, 'quant_mode': quant_mode, 'transpose_weight': transpose_weight,
    'is_interleaved': is_interleaved,
    'comm_quant_mode': comm_quant_mode, 'weight_dtype': weight_dtype, 'swiglu_quant_dtype': swiglu_quant_dtype,
    'clamp_limit': clamp_limit,
    'activation_type': activation_type, 'swiglu_alpha': swiglu_alpha, 'swiglu_beta': swiglu_beta,
    'situ_beta': situ_beta, 'situ_linear_beta': situ_linear_beta,
    'golden_ffn_chunk_size': golden_ffn_chunk_size, 'fast_random_weight': fast_random_weight,
    'golden_batch_matmul': golden_batch_matmul, 'golden_batch_matmul_min_m': golden_batch_matmul_min_m,
    'golden_expert_workers': golden_expert_workers, 'golden_log_interval': golden_log_interval,
    'golden_log_stats': golden_log_stats, 'topk_weights_type': topk_weights_type,
    'retain_npu_weights': False,
    'random_seed': random_seed,
}

cascade_graph_kwargs = {
    'loop': loop, 'is_performance': is_performance, "aic_core_num": aic_core_num, "aiv_core_num": aiv_core_num,
    "is_graph": is_graph, "graph_type": graph_type, "clamp_limit": clamp_limit,
    "local_moe_expert_num": local_moe_expert_num, "shared_expert_num": shared_expert_num,
    "expert_per_rank": expert_per_rank, "enable_shared_expert_multistream": enable_shared_expert_multistream,
    "performance_loop_outside_graph": performance_loop_outside_graph,
    "topk_weights_type": topk_weights_type,
    "weight_is_tensor_list": weight_is_tensor_list,
    "activation_type": activation_type, "swiglu_alpha": swiglu_alpha, "swiglu_beta": swiglu_beta,
    "activation_direct_output": activation_direct_output,
    "situ_beta": situ_beta, "situ_linear_beta": situ_linear_beta,
    "is_activation_params": is_activation_params,
}

def write_row_to_csv(csv_name: str, row_content: list):
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)


KERNEL_TIMING_SLOT_NAMES = (
    "dispatch",
    "group_matmul",
    "swiglu_quant",
    "gmm2",
    "combine",
)

KERNEL_TIMING_ACCUMULATED_SLOTS = {0, 1, 2, 3, 4}


def dump_critical_path_trace(output_tensor, rank, out_dir):
    """将 MegaMoe 输出中的关键路径 trace 按原始字节保存。"""
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"rank{rank}.raw")
    output_bytes = output_tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    output_bytes.tofile(raw_path)
    print(f"[TRACE] rank {rank}: raw trace saved to {raw_path}", flush=True)


def extract_kernel_timing(output_tensor, num_cores, rank, out_dir=None, cycles_to_us=50, aic_num=None):
    """Extract per-core MegaMoe timing data from the beginning of yOut."""
    if out_dir is None:
        out_dir = "."
    rank_out_dir = os.path.join(out_dir, f"rank{rank}")
    os.makedirs(rank_out_dir, exist_ok=True)

    timing_slots = 16
    timing_bytes = num_cores * timing_slots * 2 * 8
    output_bytes = output_tensor.numel() * output_tensor.element_size()
    if output_bytes < timing_bytes:
        raise RuntimeError(
            f"rank {rank}: yOut has {output_bytes} bytes, but kernel timing needs {timing_bytes} bytes"
        )

    timing = (
        output_tensor.contiguous()
        .view(torch.uint8)
        .reshape(-1)[:timing_bytes]
        .cpu()
        .view(torch.int64)
        .reshape(num_cores, timing_slots, 2)
    )
    timing_np = timing.numpy()

    raw_columns = []
    for slot_id in range(1, timing_slots + 1):
        raw_columns.extend([f"slot_{slot_id}_start", f"slot_{slot_id}_stop"])
    if aic_num is None:
        index = [f"core_{core_id}" for core_id in range(num_cores)]
    else:
        aiv_num = num_cores - aic_num
        index = (
            [f"aic_{core_id}" for core_id in range(aic_num)] +
            [f"aiv_{core_id}" for core_id in range(aiv_num)]
        )

    raw_data = timing_np.reshape(num_cores, -1).astype(np.int64)
    raw_csv_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_raw.csv")
    pd.DataFrame(raw_data, columns=raw_columns, index=index).to_csv(raw_csv_path)
    print(f"[TIMING] rank {rank}: raw table saved to {raw_csv_path}")

    start_timestamps = raw_data[:, 0::2]
    valid_origins = start_timestamps[start_timestamps != 0]
    if valid_origins.size == 0:
        raise RuntimeError(
            f"rank {rank}: no valid timing data found in yOut; check that the timing kernel is installed"
        )
    timing_origin = valid_origins.min()
    relative_data = np.where(raw_data != 0, raw_data - timing_origin, 0)
    relative_csv_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_relative.csv")
    pd.DataFrame(relative_data, columns=raw_columns, index=index).to_csv(relative_csv_path)
    print(f"[TIMING] rank {rank}: relative table saved to {relative_csv_path}")

    duration = timing_np[:, :, 1] - timing_np[:, :, 0]
    valid_duration = (timing_np[:, :, 0] != 0) & (timing_np[:, :, 1] >= timing_np[:, :, 0])
    duration_us = np.where(valid_duration, duration / cycles_to_us, 0.0)
    stage_count = len(KERNEL_TIMING_SLOT_NAMES)
    duration_columns = [f"slot_{slot_id}" for slot_id in range(1, stage_count + 1)]
    duration_csv_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_duration.csv")
    pd.DataFrame(
        duration_us[:, :stage_count], columns=duration_columns, index=index
    ).to_csv(duration_csv_path)
    print(f"[TIMING] rank {rank}: duration table saved to {duration_csv_path}")

    slot_map_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_slot_map.csv")
    pd.DataFrame({
        "slot": duration_columns,
        "stage": KERNEL_TIMING_SLOT_NAMES,
        "mode": [
            "accumulated" if slot_id in KERNEL_TIMING_ACCUMULATED_SLOTS else "interval"
            for slot_id in range(stage_count)
        ],
    }).to_csv(slot_map_path, index=False)
    print(f"[TIMING] rank {rank}: slot map saved to {slot_map_path}")

    summary_rows = []
    for slot_id, stage_name in enumerate(KERNEL_TIMING_SLOT_NAMES):
        active_core_mask = valid_duration[:, slot_id] & (duration_us[:, slot_id] > 0)
        active_core_count = int(active_core_mask.sum())
        if active_core_count == 0:
            max_core_us = 0.0
            sum_core_us = 0.0
            interval_us = 0.0
        else:
            active_durations = duration_us[:, slot_id][active_core_mask]
            max_core_us = float(active_durations.max())
            sum_core_us = float(active_durations.sum())
            interval_us = float(
                (
                    timing_np[:, slot_id, 1][active_core_mask].max() -
                    timing_np[:, slot_id, 0][active_core_mask].min()
                ) / cycles_to_us
            )
        is_accumulated = slot_id in KERNEL_TIMING_ACCUMULATED_SLOTS
        summary_rows.append({
            "slot": f"slot_{slot_id + 1}",
            "stage": stage_name,
            "mode": "accumulated" if is_accumulated else "interval",
            "active_cores": active_core_count,
            "max_core_us": max_core_us,
            "sum_core_us": sum_core_us,
            "interval_us": np.nan if is_accumulated else interval_us,
        })
    summary_csv_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"[TIMING] rank {rank}: stage summary saved to {summary_csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(
            f"[TIMING] rank {rank}: missing Python module {error.name}; "
            "CSV files were generated, charts were skipped"
        )
        return timing

    active_slots = np.any(duration_us[:, :stage_count] > 0, axis=0)
    slot_ids = np.where(active_slots)[0]
    if slot_ids.size == 0:
        print(f"[TIMING] rank {rank}: no active timing slots, charts were skipped")
        return timing

    colors = [plt.cm.tab20(slot_index % 20) for slot_index in range(len(slot_ids))]
    figure_height = max(5, num_cores * 0.35)

    fig, ax = plt.subplots(figsize=(14, figure_height))
    left = np.zeros(num_cores)
    for color_index, slot_id in enumerate(slot_ids):
        ax.barh(
            range(num_cores),
            duration_us[:, slot_id],
            left=left,
            height=0.7,
            color=colors[color_index],
            edgecolor="white",
            linewidth=0.3,
            label=f"slot_{slot_id + 1}:{KERNEL_TIMING_SLOT_NAMES[slot_id]}",
        )
        left += duration_us[:, slot_id]
    max_total = left.max()
    min_label_width = max_total * 0.01 if max_total > 0 else 0
    left = np.zeros(num_cores)
    for slot_id in slot_ids:
        for core_id in range(num_cores):
            value = duration_us[core_id, slot_id]
            if value > min_label_width:
                ax.text(
                    left[core_id] + value / 2,
                    core_id,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )
        left += duration_us[:, slot_id]
    ax.set_yticks(range(num_cores))
    ax.set_yticklabels(index, fontsize=7)
    ax.set_xlabel("time (us)")
    ax.set_title(f"Per-Core Slot Breakdown - Rank {rank} (accumulated resource time)")
    ax.invert_yaxis()
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    stack_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_stack.png")
    fig.savefig(stack_path, dpi=150)
    plt.close(fig)
    print(f"[TIMING] rank {rank}: stacked bar chart saved to {stack_path}")

    timeline_slot_ids = np.array(
        [slot_id for slot_id in slot_ids if slot_id not in KERNEL_TIMING_ACCUMULATED_SLOTS],
        dtype=np.int64,
    )
    if timeline_slot_ids.size == 0:
        return timing

    relative_timing = relative_data.reshape(num_cores, timing_slots, 2).astype(np.float64)
    relative_start_us = relative_timing[:, :, 0] / cycles_to_us
    relative_stop_us = relative_timing[:, :, 1] / cycles_to_us
    relative_duration_us = relative_stop_us - relative_start_us

    fig, ax = plt.subplots(figsize=(16, figure_height))
    timeline_colors = [plt.cm.tab20(slot_index % 20) for slot_index in range(len(timeline_slot_ids))]
    for color_index, slot_id in enumerate(timeline_slot_ids):
        ax.barh(
            range(num_cores),
            relative_duration_us[:, slot_id],
            left=relative_start_us[:, slot_id],
            height=0.7,
            color=timeline_colors[color_index],
            edgecolor="white",
            linewidth=0.3,
            label=f"slot_{slot_id + 1}:{KERNEL_TIMING_SLOT_NAMES[slot_id]}",
        )
    max_end = relative_stop_us[:, timeline_slot_ids].max()
    min_label_width = max_end * 0.01 if max_end > 0 else 0
    for slot_id in timeline_slot_ids:
        for core_id in range(num_cores):
            value = relative_duration_us[core_id, slot_id]
            if value > min_label_width:
                ax.text(
                    relative_start_us[core_id, slot_id] + value / 2,
                    core_id,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )
    ax.set_yticks(range(num_cores))
    ax.set_yticklabels(index, fontsize=7)
    ax.set_xlabel("relative time (us)")
    ax.set_title(f"Per-Core Interval Timeline - Rank {rank}")
    ax.invert_yaxis()
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    timeline_path = os.path.join(rank_out_dir, f"timing_rank_{rank}_timeline.png")
    fig.savefig(timeline_path, dpi=150)
    plt.close(fig)
    print(f"[TIMING] rank {rank}: timeline chart saved to {timeline_path}")

    return timing


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
        "shared_l1_weights", "shared_l2_weights", "shared_l1_weights_sf", "shared_l2_weights_sf",
        "w1_npu_standalone", "weight_scales1_standalone",
        "shared_l1_weights_standalone", "shared_l1_weights_sf_standalone",
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

    if activation_type not in (
        "swiglu", "swiglustep", "swigluoai", "swiglu/oai", "swiglu_oai",
        "situ", "situglu",
    ):
        raise ValueError(
            "activation_type only supports swiglu, swiglustep, swigluoai "
            f"or situglu, got {activation_type}"
        )
    if activation_type in ("situ", "situglu") and clamp_limit != -1:
        raise ValueError("situ不支持clamp_limit")

    if shared_expert_num < 0:
        raise ValueError(
            f"shared_expert_num must be >= 0, got {shared_expert_num}; "
            f"expert_per_rank={expert_per_rank}, local_moe_expert_num={local_moe_expert_num}"
        )

    if loop_acc_time < 1:
        raise ValueError(f"loop_acc_time must be >= 1, got {loop_acc_time}")

    if kernel_timing_enable:
        timing_bytes = (aic_core_num + aiv_core_num) * 16 * 2 * 8
        y_output_bytes = bs * h * torch.empty((), dtype=out_dtype).element_size()
        if y_output_bytes < timing_bytes:
            raise ValueError(
                f"kernel timing needs {timing_bytes} bytes in yOut, but bs={bs}, h={h}, "
                f"dtype={out_dtype} provide only {y_output_bytes} bytes"
            )

    if is_interleaved and (h % 256 != 0 or n % 256 != 0):
        raise ValueError(
            f"is_interleaved=True requires h and n to be 256-aligned, got h={h}, n={n}"
        )

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

PERFORMANCE_TAIL_SIZE = 60
PERFORMANCE_TAIL_SKIP = 2

TIMELINE_METRIC_BY_OP = {
    small_ops_wall_op_type: "wall_span_us",
    small_ops_resource_op_type: "resource_sum_us",
    communication_op_type: "communication_us",
    compute_op_type: "compute_us",
    moe_expert_op_type: "moe_expert_us",
    shared_expert_op_type: "shared_expert_us",
    add_op_type: "add_us",
    mega_moe_op_type: "duration_us",
}

OP_DISPLAY_NAMES = {
    small_ops_wall_op_type: "小算子真实耗时(wall)",
    small_ops_resource_op_type: "小算子耗时(sum)",
    communication_op_type: "通信耗时",
    compute_op_type: "计算耗时",
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


def _is_elementwise_mul_row(row):
    return _text_has(row, "Mul") and not _text_has(row, "Matmul")


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
    route_mask = kernel_details.apply(
        lambda row: _text_has(
            row,
            "MoeInitRouting",
            "MoeFinalizeRouting",
            "MoeDistributeDispatch",
            "MoeDistributeCombine",
            "MoeTokenPermute",
            "MoeTokenUnpermute",
        ),
        axis=1,
    )
    return set(kernel_details.loc[route_mask, "stream_id"].dropna().astype(int).tolist())


def _classify_timeline_row(row, moe_streams):
    if _text_has(row, "MegaMoe"):
        return mega_moe_op_type
    if _text_has(row, "allGather", "reduceScatter"):
        return communication_op_type
    if _text_has(row, "Add"):
        return add_op_type if shared_expert_num > 0 else None
    if _text_has(row, "MoeInitRouting", "MoeFinalizeRouting"):
        return moe_expert_op_type
    if _text_has(
        row,
        "GroupedMatmul",
        "GmmSwigluQuantV2",
        "ClippedSwiglu",
        "SwigluGroupQuant",
        "Silu",
        "Clamp",
        "DynamicMxQuant",
        "AntiMxQuant",
        "DynamicQuant",
    ) or _is_elementwise_mul_row(row):
        return moe_expert_op_type if int(row["stream_id"]) in moe_streams else shared_expert_op_type
    if _text_has(row, "ReduceSum", "Tile", "Fill"):
        return shared_expert_op_type
    if _text_has(row, "MoeDistributeDispatch", "MoeDistributeCombine"):
        return communication_op_type
    return None


def _collect_operator_samples(kernel_details, iteration_count=0):
    moe_streams = _get_moe_streams(kernel_details)
    samples = []
    for _, row in kernel_details.iterrows():
        timeline_group = _classify_timeline_row(row, moe_streams)
        if timeline_group is None:
            continue
        operator_type = str(row.get("Type", "")).strip()
        if not operator_type or operator_type.lower() == "nan":
            operator_type = str(row.get("Name", "")).strip()
        samples.append({
            "local_rank_id": int(row.get("local_rank_id", -1)),
            "timeline_group": timeline_group,
            "operator_type": operator_type,
            "duration_us": float(row["duration_us"]),
            "start_us": float(row["start_us"]),
        })
    if iteration_count > 0:
        samples.sort(key=lambda sample: sample["start_us"])
        grouped_samples = {}
        for sample in samples:
            key = (
                sample["local_rank_id"],
                sample["timeline_group"],
                sample["operator_type"],
            )
            grouped_samples.setdefault(key, []).append(sample)
        for operator_samples in grouped_samples.values():
            calls_per_iteration = max(
                1, int(round(len(operator_samples) / iteration_count))
            )
            for sample_index, sample in enumerate(operator_samples):
                sample["occurrence_index"] = sample_index % calls_per_iteration + 1
    return samples


def _operator_semantic_name(operator_type, occurrence_index):
    if "allgather" in operator_type.lower():
        allgather_inputs = {
            1: "量化X",
            2: "动态Scale",
            3: "expert_ids",
            4: "expert_scales",
        }
        return allgather_inputs.get(occurrence_index, f"输入{occurrence_index}")
    if "moeinitrouting" in operator_type.lower():
        routing_inputs = {
            1: "量化X路由",
            2: "动态Scale路由",
        }
        return routing_inputs.get(occurrence_index, f"路由{occurrence_index}")
    return ""


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
        elif _text_has(timeline.iloc[anchor_position], "MoeDistributeDispatch"):
            start_position = anchor_position
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
        communication_us = float(duration_by_group.get(communication_op_type, 0.0))
        compute_segment = segment[
            segment["timeline_group"] != communication_op_type
        ]
        compute_us = toolss_pro.calculate_interval_union_us(
            zip(compute_segment["start_us"], compute_segment["end_us"])
        )
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
            "communication_us": communication_us,
            "compute_us": compute_us,
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

def get_performance(profiling_path: str, return_dataframes=False):
    op_types = path_to_op_types[profiling_path]
    kernel_detail_paths = _get_kernel_detail_paths(profiling_path)
    timeline_rows = []
    operator_samples = []
    parsed_dataframes = []
    need_small_ops = any(op_type != mega_moe_op_type for op_type in op_types)
    need_mega_moe = mega_moe_op_type in op_types
    for kernel_detail_path in kernel_detail_paths:
        kernel_details = _read_kernel_details(kernel_detail_path)
        if kernel_details.empty:
            continue
        parsed_dataframes.append(kernel_details)
        if need_small_ops:
            small_op_iterations = _extract_small_op_iterations(
                kernel_details, kernel_detail_path
            )
            timeline_rows.extend(small_op_iterations)
        else:
            small_op_iterations = []
        if need_mega_moe:
            mega_moe_samples = _extract_mega_moe_samples(
                kernel_details, kernel_detail_path
            )
            timeline_rows.extend(mega_moe_samples)
        else:
            mega_moe_samples = []
        iteration_count = len(small_op_iterations) or len(mega_moe_samples)
        operator_samples.extend(
            _collect_operator_samples(kernel_details, iteration_count)
        )

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
    print_operator_performance_details(operator_samples)
    if return_dataframes:
        return performance_info, parsed_dataframes
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


def print_operator_performance_details(operator_samples):
    if not operator_samples:
        return

    group_display_names = {
        shared_expert_op_type: "共享专家",
        moe_expert_op_type: "MOE专家",
        communication_op_type: "通信",
        add_op_type: "Add",
        mega_moe_op_type: "融合MegaMoe",
    }
    group_order = {
        shared_expert_op_type: 0,
        moe_expert_op_type: 1,
        communication_op_type: 2,
        add_op_type: 3,
        mega_moe_op_type: 4,
    }
    grouped_samples = {}
    for sample in operator_samples:
        key = (
            sample["timeline_group"],
            sample["operator_type"],
            sample.get("occurrence_index", 1),
        )
        grouped_samples.setdefault(key, []).append(sample)

    rows = []
    sorted_items = sorted(
        grouped_samples.items(),
        key=lambda item: (
            group_order.get(item[0][0], 99),
            item[0][1].lower(),
            item[0][2],
        ),
    )
    for (timeline_group, operator_type, occurrence_index), samples in sorted_items:
        stats = _summarize_rank_metric(samples, "duration_us")
        same_type_count = sum(
            1
            for key in grouped_samples
            if key[:2] == (timeline_group, operator_type)
        )
        semantic_name = _operator_semantic_name(operator_type, occurrence_index)
        operator_display = (
            f"{operator_type} #{occurrence_index}"
            if same_type_count > 1
            else operator_type
        )
        if semantic_name:
            operator_display += f" ({semantic_name})"
        rows.append([
            group_display_names.get(timeline_group, timeline_group),
            operator_display,
            len(samples),
            format_perf_value(stats[3]),
            format_perf_value(stats[0]),
            format_perf_value(stats[1]),
            format_perf_value(stats[2]),
        ])

    print_perf_table(
        "[INFO] 算子单次调用性能统计(us)",
        [
            "分类",
            "算子类型",
            "样本数",
            "最小值的最小值",
            "中位数的最小值",
            "中位数的最大值",
            "中位数的中位数",
        ],
        rows,
    )


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
    communication = stats_by_op.get(communication_op_type, [0, 0, 0, 0])
    compute = stats_by_op.get(compute_op_type, [0, 0, 0, 0])
    mega_moe = stats_by_op.get(mega_moe_op_type, [0, 0, 0, 0])
    communication_mask_time, communication_mask_rate, compute_mask_rate = (
        toolss_pro.calculate_masking_stats(
            small_wall,
            communication,
            compute,
            mega_moe,
        )
    )
    mask_rate = toolss_pro.calculate_overall_masking_rates(
        communication_mask_rate,
        compute_mask_rate,
    )
    compare_rows = [
        ['小算子真实耗时(wall)', format_perf_value(small_wall[2])],
        ['小算子耗时(sum)', format_perf_value(small_resource[2])],
        ['融合MegaMoe', format_perf_value(mega_moe[2])],
        ['通信路径', toolss_pro.get_communication_path_label(moe_stage)],
        ['性能节省时间(us)', format_perf_value(communication_mask_time[2])],
        ['掩盖率(%)', format_perf_value(mask_rate[2])],
    ]
    print_perf_table('[INFO] 性能汇总', ['指标', '中位数的中位数'], compare_rows)
    print_perf_table(
        "[INFO] 掩盖率分项",
        ["指标", "中位数的中位数"],
        [
            ["通信掩盖率(%)", format_perf_value(communication_mask_rate[2])],
            ["计算掩盖率(%)", format_perf_value(compute_mask_rate[2])],
        ],
    )
    return communication_mask_time, mask_rate, communication_mask_rate, compute_mask_rate


def get_cascade_performance(expert_ids_by_rank=None):
    cascade_op_types = path_to_op_types[cascade_moe_profiling_path]
    mega_moe_op_types = path_to_op_types[mega_moe_profiling_path]
    cascade_moe_performance, cascade_dataframes = get_performance(
        cascade_moe_profiling_path,
        return_dataframes=True,
    )
    mega_moe_performance, mega_dataframes = get_performance(
        mega_moe_profiling_path,
        return_dataframes=True,
    )

    op_types = cascade_op_types + mega_moe_op_types
    op_performances = cascade_moe_performance + mega_moe_performance
    performance_by_type = dict(zip(op_types, op_performances))
    small_wall_stats = performance_by_type.get(
        small_ops_wall_op_type, [math.nan] * 4
    )
    mega_moe_stats = performance_by_type.get(mega_moe_op_type, [math.nan] * 4)
    communication_mask_time, mask_rate, communication_mask_rate, compute_mask_rate = print_cascade_performance(
        op_types,
        op_performances,
    )
    op_types += [
        communication_mask_time_op_type,
        mask_rate_op_type,
        communication_mask_rate_op_type,
        compute_mask_rate_op_type,
    ]
    op_performances += [
        communication_mask_time,
        mask_rate,
        communication_mask_rate,
        compute_mask_rate,
    ]
    save_performance_result('cascade', current_case_name, op_types, op_performances)
    profiling_xlsx_path = os.path.join("csv_result", "profiling.xlsx")
    try:
        report = toolss_pro.generate_profiling_xlsx(
            profiling_xlsx_path,
            current_case_name,
            moe_stage,
            shared_expert_num,
            cascade_moe_profiling_path,
            mega_moe_profiling_path,
            expert_ids_by_rank=expert_ids_by_rank,
            bs=bs,
            h=h,
            n=n,
            k=k,
            graph_type=graph_type,
            ep_world_size=ep_world_size,
            local_moe_expert_num=local_moe_expert_num,
            rank_per_dev=rank_per_dev,
            quant_dtype=quant_dtype,
            comm_quant_mode=comm_quant_mode,
            weight_dtype=weight_dtype,
            clamp_limit=clamp_limit,
            activation_type=activation_type,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            topk_weights_type=topk_weights_type,
            precomputed_metrics={
                "small_e2e_us": small_wall_stats[2],
                "mega_e2e_us": mega_moe_stats[2],
                "e2e_improvement_percent": (
                    (small_wall_stats[2] - mega_moe_stats[2])
                    / small_wall_stats[2]
                    * 100.0
                    if small_wall_stats[2] > 0 else math.nan
                ),
                "masking_rate_percent": mask_rate[2],
                "communication_masking_rate_percent": communication_mask_rate[2],
                "compute_masking_rate_percent": compute_mask_rate[2],
            },
            cascade_dataframes=cascade_dataframes,
            mega_dataframes=mega_dataframes,
        )
        print(
            f"[INFO] case={current_case_name} 性能明细已写入 "
            f"{profiling_xlsx_path}, MegaMoe E2E={report['mega_e2e_us']:.3f}us"
        )
    except Exception as exc:
        traceback.print_exc()
        print(f"[WARN] case={current_case_name} 生成 profiling.xlsx 失败: {exc}")

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

def compile_model(model, graph_pool=None):
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(1)

    print(f"------------------graph_type: {graph_type}------------------", )
    compiler_config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)

    if graph_type == 3:
        print('is_aclgraph=1, 测试场景: aclgraph入图模式')
        options = {"frozen_parameter": True}
        if graph_pool is not None:
            options["use_graph_pool"] = graph_pool
        compiled_model = torch.compile(
            model,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options=options)
    elif graph_type == 1:
        # 传统入图模式，静态shape+在线编译场景
        print('graph_type=1, 测试场景: 传统入图模式，静态shape+在线编译场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:
        # ACLNN入图模式，动态shape+二进制
        print('graph_type=2, 测试场景: ACLNN入图模式，动态shape+二进制场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)

    return compiled_model


def release_npu_model_memory():
    torch.npu.synchronize()
    if graph_type != 0:
        torch._dynamo.reset()
    gc.collect()
    torch.npu.empty_cache()

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

def prepare_single_moe_weight(weight, gmm_x_dtype_name):
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


def prepare_moe_weight(weight, gmm_x_dtype_name):
    if weight is None:
        return None
    if isinstance(weight, (tuple, list, torch.nn.ParameterList)):
        weight = torch.stack([item.npu() for item in weight], dim=0)
    return prepare_single_moe_weight(weight, gmm_x_dtype_name)


def prepare_mega_moe_weight(weight, gmm_x_dtype_name):
    if weight is None or not weight_is_tensor_list:
        return prepare_moe_weight(weight, gmm_x_dtype_name)
    weights = (
        list(weight)
        if isinstance(weight, (tuple, list, torch.nn.ParameterList))
        else list(weight.unbind(0))
    )
    return [
        prepare_single_moe_weight(
            item.clone(memory_format=torch.contiguous_format),
            gmm_x_dtype_name,
        )
        for item in weights
    ]


def prepare_npu_tensor_list(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, (tuple, list, torch.nn.ParameterList)):
        tensors = list(tensor)
    elif weight_is_tensor_list:
        tensors = [
            item.clone(memory_format=torch.contiguous_format)
            for item in tensor.unbind(0)
        ]
    else:
        tensors = [tensor]
    return [item.npu() for item in tensors]


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
        "shared_l1_weights": None,
        "shared_l2_weights": None,
        "shared_l1_weights_sf": None,
        "shared_l2_weights_sf": None,
        "weight1_standalone": None,
        "weight1_scale_standalone": None,
        "shared_l1_weights_standalone": None,
        "shared_l1_weights_sf_standalone": None,
    }
    output_names = {
        "moe": (
            "weight1",
            "weight2",
            "weight1_scale",
            "weight2_scale",
            "weight1_standalone",
            "weight1_scale_standalone",
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
    standalone_scale_outputs = {
        "weight1_scale_standalone",
        "shared_l1_weights_sf_standalone",
    }

    for chunk in iter_rank_npu_weight_chunks(ep_id, **input_kwargs):
        for output_name, chunk_name in zip(output_names[chunk["partition"]], chunk_names):
            if not is_interleaved and output_name.endswith("_standalone"):
                continue
            is_scale_output = output_name.endswith(("_scale", "_sf"))
            is_scale_output = is_scale_output or output_name in standalone_scale_outputs
            if quant_mode == 0 and is_scale_output:
                continue
            raw_value = chunk[chunk_name]
            if is_scale_output:
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

    if not is_interleaved:
        outputs["weight1_standalone"] = outputs["weight1"]
        outputs["weight1_scale_standalone"] = outputs["weight1_scale"]
    if shared_expert_num > 0:
        if not is_interleaved:
            outputs["shared_l1_weights_standalone"] = outputs["shared_l1_weights"]
            outputs["shared_l1_weights_sf_standalone"] = outputs["shared_l1_weights_sf"]
    torch.npu.synchronize()
    return outputs


def combine_cascade_weight_inputs(moe_input, shared_input, dtype_name, is_scale=False):
    restore = restore_weight_scale_input if is_scale else lambda value: restore_weight_input(value, dtype_name)
    moe_value = restore(moe_input)
    if shared_input is None:
        return moe_value
    return torch.cat((restore(shared_input), moe_value), dim=0)

def prepare_cascade_weight_partitions(weight, gmm_x_dtype_name):
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

def get_rank_x_active_mask_npu(input_data, rank):
    if is_variable_bs:
        return None
    x_active_mask = get_rank_attr(input_data, "x_active_mask", rank)
    return None if x_active_mask is None else x_active_mask.npu()

def get_token_nums_per_rank(input_data):
    if not is_variable_bs:
        return None
    token_nums = getattr(input_data, "token_nums_per_rank", None)
    if token_nums is not None:
        return tuple(int(token_num) for token_num in token_nums)
    x_by_rank = getattr(input_data, "x")
    if not isinstance(x_by_rank, (list, tuple)):
        return None
    return tuple(int(rank_x.shape[0]) for rank_x in x_by_rank)

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
        "shared_l1_weights", "shared_l2_weights", "shared_l1_weights_sf", "shared_l2_weights_sf",
        "w1_npu_standalone", "weight_scales1_standalone",
        "shared_l1_weights_standalone", "shared_l1_weights_sf_standalone",
    )
    rank_input_data = {
        field: clone_rank_attr(get_rank_attr(input_data, field, rank))
        for field in rank_fields
    }
    if is_variable_bs:
        rank_input_data["token_nums_per_rank"] = tuple(
            int(rank_x.shape[0]) for rank_x in input_data.x
        )
    return SimpleNamespace(**rank_input_data)

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
            mega_moe_kwargs.pop("shared_l1_weights", None),
            mega_moe_kwargs.pop("shared_l2_weights", None),
        )
    return model

def prepare_cascade_model_for_graph(model, cascade_kwargs):
    if should_use_static_mega_moe_weights():
        model.set_static_weights(cascade_kwargs[11], cascade_kwargs[14])
        cascade_kwargs[11] = None
        cascade_kwargs[14] = None
    return model

def get_cascade_kwargs(rank, input_data, group_ep, group_tp, prepared_weights=None):
    ep_rank_id = rank // tp_world_size
    x_npu = get_rank_attr(input_data, "x", rank).npu()
    token_nums_per_rank = get_token_nums_per_rank(input_data)
    if prepared_weights is not None:
        weight1_for_gmm = (
            prepare_moe_weight(
                prepared_weights["shared_l1_weights_standalone"], quant_dtype
            ),
            prepare_moe_weight(prepared_weights["weight1_standalone"], quant_dtype),
        )
        weight2_for_gmm = (
            prepare_moe_weight(
                prepared_weights["shared_l2_weights"], swiglu_quant_dtype
            ),
            prepare_moe_weight(prepared_weights["weight2"], swiglu_quant_dtype),
        )
        weight1_scale = prepared_weights["weight1_scale_standalone"]
        weight2_scale = prepared_weights["weight2_scale"]
        if prepared_weights["shared_l1_weights_sf_standalone"] is not None:
            weight1_scale = torch.cat(
                (
                    prepared_weights["shared_l1_weights_sf_standalone"],
                    weight1_scale,
                ),
                dim=0,
            )
            weight2_scale = torch.cat(
                (prepared_weights["shared_l2_weights_sf"], weight2_scale),
                dim=0,
            )
    else:
        w1_npu = get_rank_attr(input_data, "w1_npu_standalone", rank)
        w2_npu = get_rank_attr(input_data, "w2_npu", rank)
        weight_scales1 = get_rank_attr(input_data, "weight_scales1_standalone", rank)
        weight_scales2 = get_rank_attr(input_data, "weight_scales2", rank)
        shared_l1_weights = get_rank_attr(input_data, "shared_l1_weights_standalone", rank)
        shared_l2_weights = get_rank_attr(input_data, "shared_l2_weights", rank)
        shared_l1_weights_sf = get_rank_attr(input_data, "shared_l1_weights_sf_standalone", rank)
        shared_l2_weights_sf = get_rank_attr(input_data, "shared_l2_weights_sf", rank)
        weight1 = combine_cascade_weight_inputs(w1_npu, shared_l1_weights, quant_dtype)
        weight2 = combine_cascade_weight_inputs(w2_npu, shared_l2_weights, quant_dtype)
        weight1_scale = combine_cascade_weight_inputs(
            weight_scales1,
            shared_l1_weights_sf,
            quant_dtype,
            is_scale=True,
        )
        weight2_scale = combine_cascade_weight_inputs(
            weight_scales2,
            shared_l2_weights_sf,
            quant_dtype,
            is_scale=True,
        )
        weight1_for_gmm = prepare_cascade_weight_partitions(weight1, quant_dtype)
        weight2_for_gmm = prepare_cascade_weight_partitions(weight2, swiglu_quant_dtype)
    weight1_scale_for_gmm = None if quant_mode == 0 else weight1_scale.npu()
    weight2_scale_for_gmm = None if quant_mode == 0 else weight2_scale.npu()
    cascade_expert_ids, cascade_expert_scales, cascade_expert_num = build_cascade_expert_inputs(
        get_rank_attr(input_data, "expert_ids", rank),
        get_rank_attr(input_data, "expert_scales", rank),
    )
    cascade_kwargs = [
        x_npu, cascade_expert_ids.npu(),
        cascade_expert_scales.npu(),
        get_rank_x_active_mask_npu(input_data, rank), group_ep, group_tp, ep_rank_id, ep_world_size, cascade_expert_num, quant_mode,
        global_bs, weight1_for_gmm, out_dtype, weight1_scale_for_gmm, weight2_for_gmm,
        weight2_scale_for_gmm, dispatch_quant_out_type, comm_quant_mode, rank,
        get_torch_dtype(weight_dtype) if weight_dtype.lower() in ['float4_e2m1'] else None,
        swiglu_quant_dict[swiglu_quant_dtype],
        get_torch_dtype(quant_dtype) if quant_dtype.lower() in ['float4_e2m1'] else None
    ]
    shared_token_nums = torch.full(
        (shared_expert_num,),
        x_npu.shape[0],
        dtype=torch.int64,
        device=x_npu.device,
    )
    cascade_kwargs.append(shared_token_nums)
    if moe_stage == "prefill":
        cascade_kwargs.append(token_nums_per_rank)
    return cascade_kwargs

def get_symm_buffer_kwargs(group_ep):
    return {
        "group": group_ep,
        "num_experts": moe_expert_num,
        "num_max_tokens_per_rank": bs,
        "num_topk": k,
        "hidden": h,
        "intermediate_hidden": n // 2,
        "max_recv_token_num": bs * ep_world_size * min(local_moe_expert_num, k),
        "dispatch_quant_mode": quant_mode,
        "dispatch_quant_out_dtype": dispatch_quant_out_type,
        "combine_quant_mode": comm_quant_mode,
        "topk_weights_type": 1 if topk_weights_type else 0,
        "comm_alg": ""
        }

def get_mega_moe_kwargs(rank, input_data, distribute_buffer, prepared_weights=None):
    if prepared_weights is not None:
        weight1_for_gmm = prepare_mega_moe_weight(
            prepared_weights["weight1"], quant_dtype
        )
        weight2_for_gmm = prepare_mega_moe_weight(
            prepared_weights["weight2"], swiglu_quant_dtype
        )
        weight1_scale = prepared_weights["weight1_scale"]
        weight2_scale = prepared_weights["weight2_scale"]
    else:
        w1_npu = get_rank_attr(input_data, "w1_npu", rank)
        w2_npu = get_rank_attr(input_data, "w2_npu", rank)
        weight_scales1 = get_rank_attr(input_data, "weight_scales1", rank)
        weight_scales2 = get_rank_attr(input_data, "weight_scales2", rank)
        weight1_for_gmm = prepare_mega_moe_weight(
            restore_weight_input(w1_npu, quant_dtype), quant_dtype
        )
        weight2_for_gmm = prepare_mega_moe_weight(
            restore_weight_input(w2_npu, quant_dtype), swiglu_quant_dtype
        )
        weight1_scale = restore_weight_scale_input(weight_scales1)
        weight2_scale = restore_weight_scale_input(weight_scales2)
    weight1_for_gmm = prepare_npu_tensor_list(weight1_for_gmm)
    weight2_for_gmm = prepare_npu_tensor_list(weight2_for_gmm)
    weight1_scale_for_gmm = None if quant_mode == 0 else prepare_npu_tensor_list(weight1_scale)
    weight2_scale_for_gmm = None if quant_mode == 0 else prepare_npu_tensor_list(weight2_scale)
    mega_moe_kwargs = {
        "x": get_rank_attr(input_data, "x", rank).npu(),
        "topk_ids": get_rank_attr(input_data, "expert_ids", rank).npu(),
        "topk_weights": get_rank_attr(input_data, "expert_scales", rank).npu(),
        "l1_weights": weight1_for_gmm,
        "l2_weights": weight2_for_gmm,
        "sym_buffer": distribute_buffer,
        "l1_weights_sf": weight1_scale_for_gmm,
        "l2_weights_sf": weight2_scale_for_gmm,
        "x_active_mask": get_rank_x_active_mask_npu(input_data, rank),
        "weight1_type": dispatch_quant_out_type_dict[weight_dtype],
        "weight2_type": dispatch_quant_out_type_dict[weight_dtype],
        "rank": rank,
    }
    if shared_expert_num > 0:
        if prepared_weights is not None:
            shared_l1_weights = prepare_mega_moe_weight(
                prepared_weights["shared_l1_weights"], quant_dtype
            )
            shared_l2_weights = prepare_mega_moe_weight(
                prepared_weights["shared_l2_weights"], swiglu_quant_dtype
            )
            shared_l1_weights_sf = prepared_weights["shared_l1_weights_sf"]
            shared_l2_weights_sf = prepared_weights["shared_l2_weights_sf"]
        else:
            shared_l1_weights = prepare_mega_moe_weight(
                restore_weight_input(get_rank_attr(input_data, "shared_l1_weights", rank), quant_dtype),
                quant_dtype,
            )
            shared_l2_weights = prepare_mega_moe_weight(
                restore_weight_input(get_rank_attr(input_data, "shared_l2_weights", rank), quant_dtype),
                swiglu_quant_dtype,
            )
            shared_l1_weights_sf = restore_weight_scale_input(
                get_rank_attr(input_data, "shared_l1_weights_sf", rank)
            )
            shared_l2_weights_sf = restore_weight_scale_input(
                get_rank_attr(input_data, "shared_l2_weights_sf", rank)
            )
        shared_l1_weights = prepare_npu_tensor_list(shared_l1_weights)
        shared_l2_weights = prepare_npu_tensor_list(shared_l2_weights)
        shared_l1_weights_sf = (
            None if quant_mode == 0 else prepare_npu_tensor_list(shared_l1_weights_sf)
        )
        shared_l2_weights_sf = (
            None if quant_mode == 0 else prepare_npu_tensor_list(shared_l2_weights_sf)
        )
        mega_moe_kwargs.update({
            "shared_l1_weights": shared_l1_weights,
            "shared_l2_weights": shared_l2_weights,
            "shared_l1_weights_sf": shared_l1_weights_sf,
            "shared_l2_weights_sf": shared_l2_weights_sf,
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


def synchronize_before_mega_moe():
    torch.npu.synchronize()


def unpack_activation_debug_outputs(outputs):
    if activation_direct_output:
        return outputs[0], outputs[1], outputs[2]
    return outputs[0], outputs[1], None


def run_cascade_npu(queue, rank, input_data):
    refresh_profiling_paths()
    set_device(rank)
    (
        ep_hcomm_info_dispatch,
        tp_hcomm_info_dispatch,
        _,
        ep_group_dispatch,
        tp_group_dispatch,
        ep_group_mega,
    ) = init_hccl_comm_two_ep(rank)

    distribute_buffer = get_symm_buffer_for_mega_moe(**get_symm_buffer_kwargs(ep_group_mega))
    ep_id = rank // tp_world_size
    cascade_kwargs = None
    mega_moe_kwargs = None
    for local_owner_rank in range(rank_per_dev):
        if rank % rank_per_dev == local_owner_rank:
            print(f"[INFO] rank {rank} 开始分块重建并搬运权重", flush=True)
            rebuilt_weights = rebuild_rank_weights_to_npu(ep_id)
            cascade_group_ep = (
                ep_hcomm_info_dispatch if moe_stage == "decode" else ep_group_dispatch
            )
            cascade_group_tp = (
                tp_hcomm_info_dispatch if moe_stage == "decode" else tp_group_dispatch
            )
            cascade_kwargs = get_cascade_kwargs(
                rank,
                input_data,
                cascade_group_ep,
                cascade_group_tp,
                prepared_weights=rebuilt_weights,
            )
            mega_moe_kwargs = get_mega_moe_kwargs(
                rank,
                input_data,
                distribute_buffer,
                prepared_weights=rebuilt_weights,
            )
            del rebuilt_weights
            gc.collect()
            print(f"[INFO] rank {rank} 分块权重准备完成", flush=True)
        # dist.barrier()
    print(f'rank:{rank} 构造算子输入finished!')
    if graph_type == 0:
        cascade_moe = get_cascade_graph_model_class()(**cascade_graph_kwargs)
        if not is_performance:
            for _ in range(loop_acc_time):
                x, dispatch_out, cascade_activation_debug = unpack_activation_debug_outputs(
                    cascade_moe.forward(*cascade_kwargs)
                )
        else:
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(100, 1, 0, cascade_moe_profiling_path) as prof:
                for i in range(100):
                    x, dispatch_out, cascade_activation_debug = unpack_activation_debug_outputs(
                        cascade_moe.forward(*cascade_kwargs)
                    )
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()
                torch.npu.synchronize()

        synchronize_before_mega_moe()
        cascade_x_cpu = x.detach().cpu()
        cascade_activation_debug_cpu = (
            cascade_activation_debug.detach().cpu() if activation_direct_output else None
        )
        dispatch_out_cpu = dispatch_out[3].detach().cpu()
        del x, dispatch_out, cascade_moe, cascade_kwargs
        release_npu_model_memory()

        mega_moe_model = MegaMoeGraphModel(**cascade_graph_kwargs)
        if not is_performance:
            for _ in range(loop_acc_time):
                y, expert_token_nums, kernel_activation_debug = unpack_activation_debug_outputs(
                    mega_moe_model.forward(**mega_moe_kwargs)
                )
        else:
            with get_profiling(100, 1, 0, mega_moe_profiling_path) as prof:
               for i in range(100):
                   y, expert_token_nums, kernel_activation_debug = unpack_activation_debug_outputs(
                       mega_moe_model.forward(**mega_moe_kwargs)
                   )
                   if i == 1:
                       for _ in range(100):
                           torch.exp(tensor)
                   prof.step()
    else:
        graph_pool = torch.npu.graph_pool_handle() if graph_type == 3 else None
        if is_performance and is_graph:
            loop_time = loop + 1 if performance_loop_outside_graph else 2
        else:
            loop_time = loop_acc_time
        if is_performance:
            repeat_time = 1
            warm_time = 1
            prof = get_profiling(loop_time, repeat_time, warm_time, cascade_moe_profiling_path)
            prof.start()
    
        cascade_raw_model = prepare_cascade_model_for_graph(
            get_cascade_graph_model_class()(**cascade_graph_kwargs), cascade_kwargs
        )
        cascade_moe_model = compile_model(
            cascade_raw_model, graph_pool=graph_pool
        )
        for i in range(loop_time):
            # 调用小算子
            x, dispatch_out, cascade_activation_debug = unpack_activation_debug_outputs(
                cascade_moe_model(*cascade_kwargs)
            )

            if is_performance:
                prof.step()
        if is_performance:
            prof.stop()

        synchronize_before_mega_moe()
        cascade_x_cpu = x.detach().cpu()
        cascade_activation_debug_cpu = (
            cascade_activation_debug.detach().cpu() if activation_direct_output else None
        )
        dispatch_out_cpu = dispatch_out[3].detach().cpu()
        del x, dispatch_out, cascade_moe_model, cascade_raw_model, cascade_kwargs
        release_npu_model_memory()

        mega_moe_raw_model = prepare_mega_moe_model_for_graph(MegaMoeGraphModel(**cascade_graph_kwargs), mega_moe_kwargs)
        mega_moe_model = compile_model(
            mega_moe_raw_model, graph_pool=graph_pool
        )
        if is_performance:
            repeat_time = 1
            warm_time = 1
            prof = get_profiling(loop_time, repeat_time, warm_time, mega_moe_profiling_path)
            prof.start()
    
        for i in range(loop_time):
            # 调用融合算子
            y, expert_token_nums, kernel_activation_debug = unpack_activation_debug_outputs(
                mega_moe_model(**mega_moe_kwargs)
            )
            if is_performance:
                prof.step()
        if is_performance:
            prof.stop()
    torch.npu.synchronize()
    print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if critical_path_trace_enable:
        dump_critical_path_trace(
            y,
            rank,
            os.path.join(critical_path_trace_output_dir, current_case_name),
        )

    if kernel_timing_enable:
        timing_output = y.clone()
        try:
            extract_kernel_timing(
                timing_output,
                num_cores=aic_core_num + aiv_core_num,
                rank=rank,
                out_dir=os.path.join(kernel_timing_output_dir, current_case_name),
                cycles_to_us=kernel_timing_cycles_to_us,
                aic_num=aic_core_num,
            )
        finally:
            del timing_output

    # 获得x中的有效token
    if is_mask:
        real_tokens = get_rank_attr(input_data, "x_active_mask", rank).cpu().sum().item()
        cascade_x_cpu = cascade_x_cpu[:real_tokens, :]
        y = y[:real_tokens, :]

    x_cpu = cascade_x_cpu
    y_cpu = y.cpu()
    expert_token_nums_cpu = expert_token_nums.cpu()
    if activation_direct_output:
        kernel_activation_debug_cpu = kernel_activation_debug.detach().cpu()
        valid_activation_rows = int(expert_token_nums_cpu.to(torch.int64).sum().item())
        activation_hidden = n // 2
        kernel_activation_debug_cpu = kernel_activation_debug_cpu.reshape(-1)[
            :valid_activation_rows * activation_hidden
        ].reshape(valid_activation_rows, activation_hidden)
        golden_activation = cascade_activation_debug_cpu.reshape(-1)[
            :valid_activation_rows * activation_hidden
        ].reshape(valid_activation_rows, activation_hidden).to(torch.float32)
        kernel_activation = kernel_activation_debug_cpu.to(torch.float32)
        if golden_activation.shape != kernel_activation.shape:
            print(
                f"[ActivationDebug FAIL] shape mismatch: golden={tuple(golden_activation.shape)}, "
                f"kernel={tuple(kernel_activation.shape)}"
            )
        else:
            failed = ~torch.isclose(
                kernel_activation, golden_activation, rtol=0.005, atol=0.000025, equal_nan=True
            )
            coords = torch.nonzero(failed, as_tuple=False)
            print(f"[ActivationDebug] failed_points={coords.shape[0]}, total_points={failed.numel()}")
            abs_diff = torch.abs(kernel_activation - golden_activation)
            order = torch.argsort(abs_diff[failed], descending=True)[:activation_debug_max_print]
            for idx in order.tolist():
                row, col = coords[idx].tolist()
                print(
                    f"[ActivationDebug FAIL] row={row}, hidden={col}, "
                    f"golden={golden_activation[row, col].item():.10g}, "
                    f"kernel={kernel_activation[row, col].item():.10g}, "
                    f"abs_diff={abs_diff[row, col].item():.10g}"
                )
    del y, expert_token_nums, mega_moe_kwargs
    if graph_type == 0:
        del mega_moe_model
        if is_performance:
            del tensor
    else:
        del mega_moe_model, mega_moe_raw_model, graph_pool
    if is_performance:
        del prof
    del distribute_buffer
    release_npu_model_memory()
    queue_payload = [
        rank,
        [x_cpu, y_cpu, dispatch_out_cpu, expert_token_nums_cpu],
    ]
    queue.put(queue_payload)

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
        rank_id, rank_output = result_queue.get()
        local_rank_id = rank_id - server_index * rank_per_dev
        rank_outputs[local_rank_id] = rank_output
        pending_results -= 1

    for proc in proc_list:
        proc.join()
    queue_manager.shutdown()

    if kernel_timing_enable:
        timing_case_dir = os.path.join(kernel_timing_output_dir, current_case_name)
        print(f"[TIMING] all ranks completed; results saved to {timing_case_dir}")

    # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [
            rank_output[category_id] for rank_output in rank_outputs
        ]
        category_outputs.append(specific_category_output)

    result_mega_moe = {}
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

    res = compare_output_triangle(
        ComparedOutput.EXPERT_TOKEN_NUMS,
        "two comparison",
        golden_expert_token_nums[server_ranks],
        category_outputs[3],
        torch.float32,
        False,
    )
    result_mega_moe.update(res)
    for i in range(rank_per_dev):
        print(
            "##################Compare x in rank",
            i,
            "epId",
            i // tp_world_size,
            "tpId",
            i % tp_world_size,
            "##################",
        )
        rst_npu_x = toolss_pro.checkResult(
            category_outputs[1][i],
            golden_x[server_ranks][i],
            "npu_x",
        )
        if print_compare_details:
            rst_npu_x.print_result()
        rst_npu_golden_x = toolss_pro.checkResult(
            category_outputs[0][i],
            golden_x[server_ranks][i],
            "npu_golden_x",
        )
        if print_compare_details:
            rst_npu_golden_x.print_result()
        str1, _ = rst_npu_x.check_result(rst_npu_golden_x)
        res = "FAIL" if "error" in str1 else "PASS"
        print(f"[INFO] tensor x output 精度结果为：{res}")
        write_row_to_csv(
            "moe_combineARN_triCompare.csv",
            [current_case_name, f"output_x_rank{i}", res],
        )

    print("\n##################expert_token_nums result##################")
    print("expert_token_nums: ", result_mega_moe)
    if is_performance:
        get_cascade_performance(input_data.expert_ids)
