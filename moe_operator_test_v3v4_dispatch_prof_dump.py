# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Derived from 3fee851400a4bacf1f0e6b3d37dd08016ea86437.
# Keep the configuration/runner/CSV/trace format; only dispatch profiling is adapted.
# Offline parsing of the same script needs Python stdlib only:
#   python3 THIS_FILE --parse-only dump/*.bin --output-dir after --compare-baseline before
import sys
if __name__ == "__main__" and "--parse-only" in sys.argv:
    import argparse
    import csv
    import html
    import json
    import re
    import struct
    from collections import defaultdict, deque
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Parse/compare A5 fullmesh dispatch profiler dumps")
    parser.add_argument("--parse-only", nargs="+", required=True, metavar="BIN")
    parser.add_argument("--output-dir", type=Path, default=Path("dispatch_profiler_result"))
    parser.add_argument("--compare-baseline", type=Path)
    parser.add_argument("--cycles-per-us", type=float, default=1000.0)
    args = parser.parse_args()
    if not 0 < args.cycles_per_us < float("inf"):
        parser.error("--cycles-per-us must be positive and finite")
    source = Path(__file__).read_text(encoding="utf-8")
    # The analysis block has no imports or NPU dependencies. Use the same implementation
    # as the collection path rather than maintaining a second parser.
    analysis_start = source.index("\nCYCLES_PER_US =")
    analysis_end = source.index('\nif __name__ == "__main__":', analysis_start)
    profile_cycles_per_us = args.cycles_per_us
    exec(compile(source[analysis_start:analysis_end], __file__, "exec"), globals())
    parse_profiler_files(args.parse_only, args.output_dir)
    if args.compare_baseline:
        compare_dispatch_results(args.compare_baseline, args.output_dir)
    sys.exit(0)

import bisect
import csv
import html
import json
import re
import struct
import numpy as np
import os
import pandas as pd
import random
import socket
import sys
import time
import toolss
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu
import torchair
import traceback
from collections import defaultdict, deque
from copy import deepcopy
from enum import Enum
from itertools import chain
from math import ceil
from pathlib import Path
from quantize import torch_to_numpy_tensor, numpy_to_torch_tensor, get_dtype_int_value, numpy_float8_e5m2, numpy_float8_e4m3fn, mx_quantize, mx_dequantize, dtype_map
from torch.distributed import ReduceOp
from torch.multiprocessing import Process, Manager
from typing import List


OPS_TRANSFORMER_TORCH_EXTENSION = os.environ.get("OPS_TRANSFORMER_TORCH_EXTENSION")
if OPS_TRANSFORMER_TORCH_EXTENSION is None:
    OPS_TRANSFORMER_TORCH_EXTENSION = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ops-transformer", "torch_extension")
    )
if os.path.isdir(OPS_TRANSFORMER_TORCH_EXTENSION) and OPS_TRANSFORMER_TORCH_EXTENSION not in sys.path:
    sys.path.insert(0, OPS_TRANSFORMER_TORCH_EXTENSION)

from cann_ops_transformer.ops import MoeDistributeBuffer  # noqa: E402



# region 控制参数配置

# Device GetSystemCycle ticks per microsecond; same default as reference commit.
# Confirm against the target runtime. Raw cycle columns are always retained.
profile_cycles_per_us = 1000.0

# region 执行哪些测试
is_dispatch = 1
is_combine = 0
is_continue = 0
# endregion

# region CPU模式配置 (用于在没有NPU环境下生成golden数据)
is_cpu_only = False  # 设为True时跳过NPU算子执行，直接返回golden数据
# endregion

# region 性能检测相关

is_performance = 0
loop = 105  # 性能检测时的性能采集轮数
repeat = 1  # 性能采集分为几个批次
warmup = 0  # 每批次的热身轮数

# endregion

# region 特性使能
comm_alg = 'fullmesh_v2'
dispatch_comm_alg = comm_alg
combine_comm_alg = "" if comm_alg in ["fullmesh_v1", "fullmesh_v2"] else comm_alg
is_ccu = (comm_alg == "ccu")
is_fullmesh_v2 = (comm_alg == "fullmesh_v2")
graph_type = 1  # 0为单算子，1为静态图模式，2为动态图模式（当前不支持），3为aclgraph模式
use_graph_cache = False  # 是否使能graph_cache
enable_static_kernel_compile = True  # graph_type=3时是否使能静态Kernel编译

# 0: dispatch算子非量化场景
# 1：静态量化
# 2：PerToken动态量化
# 3：PerGroup动态量化
# 4：MX量化
# 5：MX Clip量化
quant_mode = 0
comm_quant_mode = 0
combine_mx_quant = (comm_quant_mode == 3 or comm_quant_mode == 4)
x_active_mask_dim = 0
is_2d_mask = 1 if x_active_mask_dim == 2 else 0
is_mask = 1 if x_active_mask_dim == 1 else 0
is_variable_bs = 0
is_variable_local_moe_expert_num = 0  # 是否使能可变moe卡上专家数
is_shared_expert_x = 1
is_quant_scales = 1
# expertScale入参模式：0-有效Tensor，1-None，2-一维空Tensor
dispatch_expertscale = 0
combine_expertscale = 0
# 调用路径：1-low_latency接口，0-torch_npu接口
is_lowlatency = 0
buff_size_lowlatency = 0
is_full_core = True  # True时为全核模式，使用全部的核，False时使用部分核
expert_shard_type = 0  # 0 共享专家部署在最前面的卡上，仅支持 0
core_num = 16 if not is_full_core else 32  # 非全核模式使用的核数（单个核上有一个aic核，一个aiv核）

is_quant = (quant_mode > 0)
is_unfixed_real_token = is_mask or is_variable_bs  # 每张卡上真实token数是否固定
aic_core_num = core_num  # 非全核模式使用的aic核数
aiv_core_num = core_num * 2  # 非全核模式使用的aiv核数

# 控制dispatch输出的expert_token_nums的格式：
# 0表示每个专家收到token数量的前缀和；
# 1表示每个专家收到的token数量（默认）
expert_token_nums_type = 1

is_elastic = 0  # 是否动态缩容      ############ 多机环境仅支持缩容单机，不支持调节下列选项
elastic_shared_ranks = 0  # 共享专家缩容卡数
elastic_moe_ranks = 0  # moe专家卡缩容卡数
elastic_random_ranks = 0  # 随机缩容卡数
elastic_shared = (elastic_shared_ranks > 0)  # 是否缩共享专家卡
elastic_moe = (elastic_moe_ranks > 0)  # 是否缩moe专家
elastic_random = (elastic_random_ranks > 0)  # 是否随机缩容

# 性能打点模式相关
is_performance_info = 0
specified_slow_rank = [0]
random_slow_rank_num = 3
slow_rank_time = 4

zero_expert_num = 0
copy_expert_num = 0
const_expert_num = 244
is_zero_compute_expert = zero_expert_num + copy_expert_num + const_expert_num != 0
# endregion

# region 通信域配置

server_num = 1  # 单机或者多机模式
tp_world_size = 1  # 1或2, 2时dispatch算子在tp域执行allgather, combine算子执行reducescatter, 1时均不执行
shared_expert_num = 0
shared_expert_rank_num = 0
local_moe_expert_num = 24
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
os.environ["LOCAL_WORLD_SIZE"] = str(rank_per_dev)


world_size = server_num * rank_per_dev  # 通信域总die数
ep_world_size = world_size // tp_world_size
rank_num_per_shared_expert = (shared_expert_rank_num // shared_expert_num
                                if shared_expert_num > 0
                                else 0)  # 每个共享专家占据的卡数
moe_rank_num = ep_world_size - shared_expert_rank_num  # moe专家卡数
moe_expert_num = moe_rank_num * local_moe_expert_num  # moe专家数

is_shared = (shared_expert_rank_num > 0)
shared_expert_rank_num_ela = shared_expert_rank_num if (not is_elastic or (shared_expert_num == 0) or (elastic_shared_ranks % shared_expert_num) or (not elastic_shared) or (elastic_shared_ranks == shared_expert_rank_num)) else (shared_expert_rank_num - elastic_shared_ranks)
rank_num_per_shared_expert_ela = shared_expert_rank_num_ela // shared_expert_num if shared_expert_num > 0 else 0

is_elastic_info = (is_elastic or elastic_moe_ranks or elastic_shared_ranks or elastic_random_ranks) and (tp_world_size == 1)

server_ranks = slice(server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev) # 本机各die索引

include_tp_info = tp_world_size >= 2   # 新接口仅通过tp_world_size/tp_rank_id表达TP域
# endregion

# region 输入输出维度相关

def cal_hccl_buffsize(bs, h, k, ep_world_size, local_moe_expert_num, shared_expert_num):
    MAX_OUT_DTYPE_SIZE = 2
    WIN_ADDR_ALIGN = 512
    FULL_MESH_DATA_ALIGN = 480
    UB_ALIGN = 32
    SCALE_EXPAND_IDX_BUFFER = 44
    DOUBLE_DATA_BUFFER = 2
    EP_RANK_OFFSET_STEP = 8192
    ep_rank_offset = ep_world_size * EP_RANK_OFFSET_STEP

    token_need_size_combine = ((h * MAX_OUT_DTYPE_SIZE + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    token_actual_len = ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1) // UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER
    if is_fullmesh_v2:
        token_need_size_dispatch = ((token_actual_len + FULL_MESH_DATA_ALIGN - 1) // FULL_MESH_DATA_ALIGN) * WIN_ADDR_ALIGN
    else:
        token_need_size_dispatch = ((token_actual_len + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    actual_size = ((bs * token_need_size_dispatch * ep_world_size * local_moe_expert_num)
                    + (bs * token_need_size_combine * (k + shared_expert_num))) * DOUBLE_DATA_BUFFER \
                   + ep_rank_offset

    hccl_buffsize = ceil(actual_size / 1024 / 1024 / 2) + 5 # 留出 5MB 空间余量
    print(f"{hccl_buffsize=}")
    return str(hccl_buffsize)

bs = 120
h = 6351
scales_h = 32  # 非量化直转hifp8场景2中scales的第二维长度，需<=h
# k = moe_expert_num if moe_expert_num < 8 else 8    # 每个token发给多少个moe专家
k = 8

if is_lowlatency:
    hccl_buffsize = cal_hccl_buffsize(
        bs, h, k, ep_world_size, local_moe_expert_num, shared_expert_num
    )
    buff_size_lowlatency = int(hccl_buffsize) * 1024 * 1024 * 2
    print(f'{buff_size_lowlatency=}')
    # exit()
else:
    # 计算HCCL_BUFFERSIZE大小
    os.environ['HCCL_BUFFSIZE'] = cal_hccl_buffsize(bs, h, k, ep_world_size, local_moe_expert_num, shared_expert_num)

global_bs = bs * ep_world_size
max_shared_group_num = ceil(ep_world_size / rank_num_per_shared_expert) if (rank_num_per_shared_expert > 0) else 0
A_shared = bs * max_shared_group_num
A_moe = global_bs * min(local_moe_expert_num, k)
if is_elastic or elastic_moe_ranks or elastic_shared_ranks or elastic_random_ranks:
    A_shared = A_moe = max(A_shared, A_moe)

# endregion

# region 产生什么样的输入

# 静态量化场景下，是不是所有专家共享scales
# 0：所有专家不共享scales
# 1：所有专家共享scales，且维度为(h,)
# 2：所有专家共享scales，且维度为(1,)
is_static_shared_scales = 0

x_dtype = 'fp8_e4m3fn'
expandx_dtype = 'fp8_e4m3fn'
scales_dtype = "fp8_e8m0" if dtype_map.get(expandx_dtype, "").startswith("float4") else "float"
is_topk_balance = False  # 是否生成均衡的topK
is_x_random = True  # 是否让x值随机
random_seed = 1099

input_dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e5m2": torch.float8_e5m2,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "hifloat8": torch.uint8,
    "float4_e2m1": torch.uint8,
    "float4_e1m2": torch.uint8,
}.get(dtype_map.get(x_dtype, ""), torch.float16)
print(f"输入x类型为：{x_dtype}，对应的Tensor类型：{input_dtype}")
if input_dtype in (torch.bfloat16, torch.float16):
    combine_input_dtype = input_dtype
else:
    torch.manual_seed(random_seed)
    combine_input_dtype = torch.float16 if torch.rand(1) < 0.5 else torch.bfloat16
    # combine_input_dtype = torch.float16

# 非量化场景下特殊类型直转/转发相关标志
NOQ_FORWARD_DTYPES = {"hifloat8", "float8_e5m2", "float8_e4m3fn", "float4_e2m1", "float4_e1m2"}
is_noq_cast_hifp8 = (quant_mode == 0 and dtype_map.get(expandx_dtype, "") == "hifloat8"
                     and dtype_map.get(x_dtype, "") in ("float16", "bfloat16"))
is_noq_forward = (quant_mode == 0
                  and dtype_map.get(expandx_dtype, "") in NOQ_FORWARD_DTYPES
                  and dtype_map.get(x_dtype, "") == dtype_map.get(expandx_dtype, ""))
is_noq_special = is_noq_cast_hifp8 or is_noq_forward


# endregion

# region 测试脚本执行相关

is_continue_golden_expand_x = True  # 在级联模式下，是否将dispatch的golden_expand_x直接传给combine算子
is_save_tensor_to_bin = False
fullmesh_v2_profiler_size = 115264
is_check_setting = True
current_case_name = 'MOE_Dis_Com_2_bs120_h6351_k8_xdtypefp8_e4m3fn_quant0_scale1_continue1_tp1_localmoe24_shareRank0_moeExpert48_mte_ID0049'


# endregion

# endregion

# region 常用常量

# region 精度对比相关常量

class ComparedOutput(Enum):
    EP_RECV_COUNTS = "ep_recv_counts"
    TP_RECV_COUNTS = "tp_recv_counts"
    EXPAND_X = "expand_x"
    EXPAND_IDX = "expand_idx"
    EXPAND_SCALES = "expand_scales"
    DYNAMIC_SCALES = "dynamic_scales"
    EXPERT_TOKEN_NUMS = "expert_token_nums"
    X = "x"


# endregion

# region 性能处理相关常量

dispatch_profiling_path = f'./Profiling_Result_{current_case_name}_dispatch'
combine_profiling_path = f'./Profiling_Result_{current_case_name}_combine'
continue_profiling_path = f'./Profiling_Result_{current_case_name}_continue'
dispatch_op_type = 'MoeDistributeDispatch'
combine_op_type = 'MoeDistributeCombine'
path_to_op_types = {
    dispatch_profiling_path: [dispatch_op_type],
    combine_profiling_path: [combine_op_type],
    continue_profiling_path: [dispatch_op_type, combine_op_type]
}
# profiling 批次判定窗口：只统计与最新 op_summary 文件修改时间相差不超过该秒数的文件，避免历史结果污染。
profiling_file_time_window_seconds = 30

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

# 二叉树累加
def binary_tree_sum(tensor):
    if tensor.numel() == 1:
        return tensor
    elif tensor.numel() % 2 == 1:
        # 如果元素个数是奇数，先在末尾添加一个0
        tensor = torch.cat([tensor, torch.zeros([1], dtype=tensor.dtype)])
    # 计算每对元素的和
    new_tensor = tensor.view(-1, 2).sum(dim=1)

    # 递归调用
    return binary_tree_sum(new_tensor)

def concat_tensors(tensors, dim=0):
    """
    主要用于处理float8_e4m3fn，float8_e5m2这些类型的拼接，这些类型torch可能不支持
    """
    valid_index = 0
    while(valid_index < len(tensors) and tensors[valid_index].shape[0] == 0):
        valid_index += 1
    if(valid_index >= len(tensors)):
        return tensors[-1]
    dtype = tensors[valid_index].dtype
    dim = dim % tensors[valid_index].ndim

    for t in tensors:
        if t.shape[0] == 0:
            continue
        if t.dtype != dtype:
            raise ValueError(f"dtype 不一致：{t.dtype} != {dtype}")
        if any(t.shape[d] != tensors[0].shape[d] for d in range(t.ndim) if d != dim):
            raise ValueError(f"非拼接维度不匹配：{tensors[0].shape} vs {t.shape}")

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

def bool_index_tensor(tensor, bool_mask):
    """
    处理 float8_e5m2 / float8_e4m3fn 等不支持布尔索引的类型。
    float8 每个元素 1 字节，与 uint8 相同，可安全 view 后索引再 view 回去。
    """
    if tensor.dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        original_dtype = tensor.dtype
        return tensor.view(torch.uint8)[bool_mask].view(original_dtype)
    return tensor[bool_mask]


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
    # hifloat8用的torch.uint8存储
    expand_x_numpy = expand_x.cpu().numpy().view(hifloat8).astype(np.float32)
    return torch.from_numpy(expand_x_numpy).to(device=expand_x.device, dtype=target_type)

def convert_expand_x_dtype(expand_x):
    """
    将dispatch输出的expand_x转为fp16或bf16，用于combine输入
    """
    if expand_x.dtype in (torch.float16, torch.bfloat16):
        return expand_x

    from quantize import dtype_map
    mapped_dtype = dtype_map.get(expandx_dtype, None)
    if mapped_dtype in ("float8_e4m3fn", "float8_e5m2", "int8", "float16", "bfloat16"):
        # torch原生支持的类型
        return expand_x.to(combine_input_dtype)
    elif mapped_dtype == "hifloat8":
        # return _convert_hifloat8_to_target_type(expand_x, combine_input_dtype)
        return expand_x.to(combine_input_dtype)
    elif mapped_dtype in ("float4_e2m1", "float4_e1m2"):
        # fp4: packed uint8 [N, h//2]，需解包为 [N, h] fp16/bf16
        from quantize import trans_np_fp4_e2m1_tensor_to_bfloat16_v2, trans_np_fp4_e1m2_tensor_to_bfloat16_v2
        np_uint8 = expand_x.cpu().numpy()
        if mapped_dtype == "float4_e2m1":
            fp32_np = trans_np_fp4_e2m1_tensor_to_bfloat16_v2(np_uint8)
        else:
            fp32_np = trans_np_fp4_e1m2_tensor_to_bfloat16_v2(np_uint8)
        return torch.from_numpy(fp32_np).to(device=expand_x.device, dtype=combine_input_dtype)
    else:
        raise ValueError(f"Unsupported expand_x dtype {expandx_dtype}")

def convert_combine_output_to_input_dtype(combine_output):
    """
    将combine输出(fp16/bf16)转回dispatch所需的输入类型，用于连续性能测试中将combine结果作为下一次dispatch输入。
      - fp16/bf16 → 直接返回
      - fp8 (torch原生) → .to(input_dtype)
      - hifloat8 → float32 numpy astype(hifloat8) view uint8，shape不变 [N, H]
      - fp4 → float32 numpy 打包为packed uint8，shape [N, H] → [N, H//2]
    """
    if combine_output.dtype == input_dtype:
        return combine_output

    from quantize import dtype_map
    mapped_dtype = dtype_map.get(x_dtype, None)
    if mapped_dtype in ("float8_e4m3fn", "float8_e5m2"):
        return combine_output.to(input_dtype)
    elif mapped_dtype == "hifloat8":
        return combine_output.to(torch.uint8)
    elif mapped_dtype in ("float4_e2m1", "float4_e1m2"):
        # fp4以packed uint8存储：两个fp4合一个uint8，shape最后一维减半 [N, H] → [N, H//2]
        H = combine_output.shape[-1]
        return combine_output[...,:H//2].to(torch.uint8).contiguous()
    else:
        raise ValueError(f"Unsupported x_dtype {x_dtype} for combine→dispatch conversion")

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


def dump_fullmesh_v2_profiler(expand_x, rank):
    if comm_alg != "fullmesh_v2":
        return
    raw = expand_x.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    if raw.numel() < fullmesh_v2_profiler_size:
        raise RuntimeError(
            f"expand_x has only {raw.numel()} bytes, profiler needs {fullmesh_v2_profiler_size} bytes"
        )
    host = raw[:fullmesh_v2_profiler_size].numpy()
    magic = int.from_bytes(host[:4].tobytes(), byteorder="little")
    if magic != 0x50524F46:
        raise RuntimeError(f"invalid FullMeshV2 profiler magic: 0x{magic:08x}")
    path = f"fullmesh_v2_prof_rank{rank}.bin"
    host.tofile(path)
    print(f"[INFO] FullMeshV2 profiler saved to {os.path.abspath(path)}", flush=True)


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
    print(f"{dispatch_expertscale=}")
    print(f"{combine_expertscale=}")
    print(f"{is_lowlatency=}")


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

    if k > (moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num):
        error_log_and_exit("k 不能大于 moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num")

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

    if x_dtype not in dtype_map:
        error_log_and_exit(f"不支持的x_dtype:{x_dtype}")

    if expandx_dtype not in dtype_map:
        error_log_and_exit(f"不支持的expandx_dtype:{expandx_dtype}")

    if quant_mode == 0:
        if dtype_map.get(expandx_dtype, "") in NOQ_FORWARD_DTYPES:
            expand_x_mapped = dtype_map.get(expandx_dtype, "")
            x_mapped = dtype_map.get(x_dtype, "")
            # 场景1: bf16/fp16输入 → 直转hifp8（仅hifloat8支持）
            if expand_x_mapped == "hifloat8" and x_mapped in ("float16", "bfloat16"):
                pass
            # 场景2: 同类型输入输出 + per-token scales（支持所有NOQ_FORWARD_DTYPES）
            elif x_mapped == expand_x_mapped:
                if not is_quant_scales:
                    error_log_and_exit(f"quant_mode=0且x_dtype为{x_dtype}时，必须传入scales（is_quant_scales=1）")
                if scales_h > h:
                    error_log_and_exit("scales_h不能大于h")
                if expand_x_mapped.startswith("float4") and h % 2 != 0:
                    error_log_and_exit("fp4场景下h必须为偶数")
            else:
                error_log_and_exit(f"quant_mode=0且expandx_dtype为{expandx_dtype}时，x_dtype须为fp16/bf16或与expandx_dtype相同")
        elif dtype_map[expandx_dtype] != dtype_map[x_dtype]:
            error_log_and_exit("对于dispatch量化模式0，x_dtype必须和expandx_dtype保持一致")

    if quant_mode in (4, 5) and dtype_map.get(expandx_dtype, "").startswith("float4") and h % 2 != 0:
        error_log_and_exit("mxfp4量化场景下，h值必须为偶数")

    if (quant_mode == 1 and dtype_map[expandx_dtype] not in ("int8", "hifloat8")) or \
       (quant_mode == 2 and dtype_map[expandx_dtype] not in ("int8", "float8_e5m2", "float8_e4m3fn")) or \
       (quant_mode == 3 and dtype_map[expandx_dtype] not in ("float8_e5m2", "float8_e4m3fn")) or \
       (quant_mode in (4, 5) and dtype_map[expandx_dtype] not in ("float8_e5m2", "float8_e4m3fn", "float4_e2m1", "float4_e1m2")):
        error_log_and_exit(f"对于dispatch量化模式{quant_mode}，expand_x类型{expandx_dtype}不支持")

    if graph_type == 2:
        error_log_and_exit("当前不支持动态图模式，graph_type合法值为0、1和3")
    if graph_type not in (0, 1, 3):
        error_log_and_exit("graph_type合法值为0、1和3")

    if dispatch_expertscale not in (0, 1):
        error_log_and_exit("dispatch_expertscale合法值为0、1")
    if combine_expertscale not in (0, 1, 2):
        error_log_and_exit("combine_expertscale合法值为0、1和2")
    if is_lowlatency not in (0, 1):
        error_log_and_exit("is_lowlatency合法值为0和1")

    if is_lowlatency == 0 and combine_expertscale == 1:
        error_log_and_exit("torch_npu路径的combine不支持expertScales为None，请使用模式0或2")

    if is_lowlatency == 1 and tp_world_size != 1:
        error_log_and_exit("DeepEP low_latency接口测试脚本仅支持 tp_world_size=1")

    if is_lowlatency == 1 and is_performance_info:
        error_log_and_exit("DeepEP low_latency接口不支持 performance_info 入参")

    if comm_quant_mode == 1:
        error_log_and_exit("当前不支持 int12 通信量化, comm_quant_mode 合法值仅有 0、2、3、4")

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

    if elastic_random and (elastic_moe or elastic_shared):
        error_log_and_exit("随机缩容不和moe缩容、shared缩容同时开")

    if graph_type == 0 and is_full_core == 0:
        error_log_and_exit("单算子不支持控核")

    if ((rank_per_dev * server_num - shared_expert_rank_num_ela - elastic_shared_ranks - elastic_moe_ranks - elastic_random_ranks) * local_moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num) < k:
        error_log_and_exit("缩容后的moe专家数小于K，无法生成expert_ids")

    if is_ccu and is_2d_mask:
        error_log_and_exit("CCU通信方式不支持2d_mask")

# endregion

# region 性能数据处理函数

def get_latest_profiling_file_infos(profiling_path: str, file_prefix: str):
    """
    按文件修改时间获取最近一次 profiling 产生的文件。

    只保留与最新文件时间差不超过 profiling_file_time_window_seconds 的文件，
    避免在没有清理 Profiling 目录时统计到历史结果。
    """
    file_infos = []
    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if not filename.startswith(file_prefix):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                file_mtime = os.path.getmtime(file_path)
            except OSError as err:
                print(f"[WARNING] 获取 profiling 文件时间失败，已跳过: {file_path}, {err}")
                continue
            file_infos.append((file_mtime, file_path))

    if not file_infos:
        print(f"[WARNING] 未找到 {profiling_path} 下的 {file_prefix}* 文件")
        return []

    file_infos.sort(key=lambda info: info[0], reverse=True)
    latest_mtime = file_infos[0][0]
    return [
        file_info
        for file_info in file_infos
        if latest_mtime - file_info[0] <= profiling_file_time_window_seconds
    ]

def get_performance(profiling_path: str):
    def get_execution_time(profiling_path: str):
        op_types = path_to_op_types[profiling_path]
        rank_types = ['shared', 'moe']
        performance_info = {(op_type, rank_type): [] for op_type in op_types for rank_type in rank_types}

        long_wait_ranks = set()
        selected_rank_ids = set()
        for _, op_summary_path in get_latest_profiling_file_infos(profiling_path, "op_summary"):
            if len(selected_rank_ids) >= rank_per_dev:
                break

            try:
                op_summary = pd.read_csv(op_summary_path)
                local_rank_id = int(op_summary['Device_id'].iloc[0])
            except Exception as err:
                print(f"[WARNING] 读取 op_summary 文件失败，已跳过: {op_summary_path}, {err}")
                continue

            if local_rank_id in selected_rank_ids:
                continue
            selected_rank_ids.add(local_rank_id)

            ep_rank_id = (rank_per_dev * server_index + local_rank_id) // tp_world_size
            rank_type = 'shared' if ep_rank_id < shared_expert_rank_num else 'moe'
            for op_type in op_types:
                # 模糊前缀匹配：兼容 'MoeDistributeDispatch'、'MoeDistributeDispatchV2/V3' 等版本后缀
                op_info = op_summary[op_summary['OP Type'].astype(str).str.startswith(op_type)]

                op_execute_time_series = op_info['Task Duration(us)'].tail(50)
                performance_info[(op_type, rank_type)].append(op_execute_time_series)

                mean_op_wait_time = op_info['Task Wait Time(us)'].tail(50).mean()
                if mean_op_wait_time > 5:
                    long_wait_ranks.add(local_rank_id)

        if len(selected_rank_ids) < rank_per_dev:
            print(
                f"[WARNING] 仅统计到 {len(selected_rank_ids)}/{rank_per_dev} 个最新 op_summary 文件，"
                f"已忽略与最新文件时间差超过 {profiling_file_time_window_seconds} 秒的历史结果"
            )

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

    if is_quant or is_noq_special:
        if dtype_map.get(expandx_dtype, "").startswith("float4"):
            expand_x_dtype_bytes = 0.5  # fp4: 2元素packed为1字节
        else:
            expand_x_dtype_bytes = 1
    else:
        expand_x_dtype_bytes = 2
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
    print('[INFO] Dispatch算子内存占用:')
    print_dataframe(memory_usage)


def print_combine_performance(combine_performance):
    (mean_share_performance, mean_moe_performance, min_share_performance, min_moe_performance), memory_usage = combine_performance

    print(f'[INFO] MC2融合算子Average性能为: combine共享:{mean_share_performance} us, combine Moe:{mean_moe_performance} us\n')
    print(f'[INFO] MC2融合算子Min性能为: combine共享:{min_share_performance} us, combine Moe:{min_moe_performance} us\n')
    print('[INFO] Combine算子内存占用:')
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
    print('[INFO] Dispatch与Combine算子总内存占用:')
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

def get_aclgraph_compile_options():
    if enable_static_kernel_compile:
        return {"static_kernel_compile": True, "disable_static_kernel_compile_cache": False}
    return None

def use_aclgraph_limit_core():
    return graph_type == 3 and not is_full_core

def call_with_aclgraph_limit_core(func, *args, **kwargs):
    if use_aclgraph_limit_core():
        with torch.npu.npugraph_ex.scope.limit_core_num(aic_core_num, aiv_core_num):
            return func(*args, **kwargs)
    return func(*args, **kwargs)

def aclgraph_limited_inner_forward(self, *args, **kwargs):
    return call_with_aclgraph_limit_core(self.inner_forward, *args, **kwargs)

def get_aclgraph_cache_func(func):
    if not use_aclgraph_limit_core():
        return func
    module = getattr(func, "__self__", None)
    if module is None:
        return func
    bound_method = aclgraph_limited_inner_forward.__get__(module, module.__class__)
    setattr(module, "aclgraph_limited_inner_forward", bound_method)
    return bound_method

def cache_compile_model(func, cache_dir, config, backend, dynamic, options=None):
    os.makedirs(cache_dir, exist_ok=True)
    if graph_type == 3:
        cache_func = get_aclgraph_cache_func(func)
        cache_kwargs = {"cache_dir": cache_dir, "dynamic": dynamic}
        if options is not None:
            cache_kwargs["options"] = options
        return torch.npu.npugraph_ex.inference.cache_compile(cache_func, **cache_kwargs)

    import torchair
    return torchair.inference.cache_compile(
        func, cache_dir=cache_dir, config=config, backend=backend, ge_cache=True, dynamic=dynamic, fullgraph=True)

class ACLGraphLimitCoreWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return call_with_aclgraph_limit_core(self.model, *args, **kwargs)

def compile_model(model):
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(1)

    print(f"------------------graph_type: {graph_type}------------------", )

    if graph_type == 1:
        from torchair.configs.compiler_config import CompilerConfig

        if is_full_core:
            compiler_config = None
        else:
            compiler_config = CompilerConfig()
            compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
            print(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")

        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        # 传统入图模式，静态shape+在线编译场景
        print('graph_type=1, 测试场景: 传统入图模式，静态shape+在线编译场景')
        if use_graph_cache:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compile_cache")
            model.get_cache_model(cache_dir, compiler_config, npu_backend, dynamic=False)
            compiled_model = model
        else:
            compiled_model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    elif graph_type == 3:
        print('graph_type=3, 测试场景: aclgraph模式，静态shape+npugraph_ex场景')
        options = get_aclgraph_compile_options()
        if enable_static_kernel_compile:
            print('启用静态Kernel编译')
        if use_aclgraph_limit_core():
            print(f"aclgraph控核: {aic_core_num} aic核, {aiv_core_num} aiv核")
        if use_graph_cache:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compile_cache")
            model.get_cache_model(cache_dir, None, "npugraph_ex", dynamic=False, options=options)
            compiled_model = model
        else:
            compile_target = ACLGraphLimitCoreWrapper(model) if use_aclgraph_limit_core() else model
            compiled_model = torch.compile(
                compile_target, backend="npugraph_ex", dynamic=False, fullgraph=True, options=options)
    else:
        print(f"不支持的graph_type: {graph_type}")
        exit(1)

    return compiled_model


class MOE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cache_model = None

    def forward(self, *args, **kwargs):
        if use_graph_cache:
            return self.cache_model(*args, **kwargs)
        return self.inner_forward(*args, **kwargs)

    def get_cache_model(self, cache_dir, config, backend, dynamic, options=None):
        self.cache_model = cache_compile_model(self.inner_forward, cache_dir, config, backend, dynamic, options)


class MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(MOE_GRAPH_Model):
    def __init__(self, group_ep):
        super().__init__()
        print("Init group_ep")
        self.group_ep = group_ep
        self.distribute_buffer = MoeDistributeBuffer(group_ep, buff_size_lowlatency) if is_lowlatency == 1 else None

    def inner_forward(
        self, x, expert_ids, expert_scales, x_active_mask, group_ep, group_tp, ep_rank_id, tp_rank_id,
        ep_world_size, tp_world_size, expert_shard_type, shared_expert_num, shared_expert_rank_num,
        moe_expert_num, scales, quant_mode, global_bs, elastic_info, expert_token_nums_type,
        zero_expert_num, copy_expert_num, const_expert_num, comm_alg, y_dtype=None, x_dtype=None, scales_dtype=None
    ):
        dispatch_kwargs = {
            'x': x, 'expert_ids': expert_ids, 'expert_scales': expert_scales,
            'x_active_mask': x_active_mask, 'group_ep': group_ep, 'group_tp': group_tp,
            'ep_rank_id': ep_rank_id, 'tp_rank_id': tp_rank_id, 'ep_world_size': ep_world_size,
            'tp_world_size': tp_world_size, 'expert_shard_type': expert_shard_type,
            'shared_expert_num': shared_expert_num, 'shared_expert_rank_num': shared_expert_rank_num,
            'moe_expert_num': moe_expert_num, 'scales': scales, 'quant_mode': quant_mode,
            'global_bs': global_bs, 'elastic_info': elastic_info,
            'expert_token_nums_type': expert_token_nums_type, 'zero_expert_num': zero_expert_num,
            'copy_expert_num': copy_expert_num, 'const_expert_num': const_expert_num,
            'comm_alg': comm_alg, 'y_dtype': y_dtype, 'x_dtype': x_dtype, 'scales_dtype': scales_dtype,
        }
        return call_low_latency_dispatch(self.distribute_buffer, dispatch_kwargs)


class MOE_LOW_LATENCY_COMBINE_GRAPH_Model(MOE_GRAPH_Model):
    def __init__(self, group_ep):
        super().__init__()
        print("Init group_ep")
        self.group_ep = group_ep
        self.distribute_buffer = MoeDistributeBuffer(group_ep, buff_size_lowlatency) if is_lowlatency == 1 else None

    def inner_forward(
        self, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, tp_send_counts, expert_scales,
        x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
        group_ep, group_tp, ep_rank_id, tp_rank_id, ep_world_size, tp_world_size, expert_shard_type,
        shared_expert_num, shared_expert_rank_num, moe_expert_num, comm_quant_mode, global_bs,
        elastic_info, zero_expert_num, copy_expert_num, const_expert_num, comm_alg, expand_scales=None
    ):
        combine_kwargs = {
            'expand_x': expand_x, 'expert_ids': expert_ids,
            'assist_info_for_combine': assist_info_for_combine, 'ep_send_counts': ep_send_counts,
            'tp_send_counts': tp_send_counts, 'expert_scales': expert_scales,
            'x_active_mask': x_active_mask, 'expand_scales': expand_scales,
            'shared_expert_x': shared_expert_x, 'ori_x': ori_x,
            'const_expert_alpha_1': const_expert_alpha_1, 'const_expert_alpha_2': const_expert_alpha_2,
            'const_expert_v': const_expert_v, 'group_ep': group_ep, 'group_tp': group_tp,
            'ep_rank_id': ep_rank_id, 'tp_rank_id': tp_rank_id, 'ep_world_size': ep_world_size,
            'tp_world_size': tp_world_size, 'expert_shard_type': expert_shard_type,
            'shared_expert_num': shared_expert_num, 'shared_expert_rank_num': shared_expert_rank_num,
            'moe_expert_num': moe_expert_num, 'comm_quant_mode': comm_quant_mode,
            'global_bs': global_bs, 'elastic_info': elastic_info,
            'zero_expert_num': zero_expert_num, 'copy_expert_num': copy_expert_num,
            'const_expert_num': const_expert_num, 'comm_alg': comm_alg,
        }
        return call_low_latency_combine(self.distribute_buffer, combine_kwargs)


class MOE_LOW_LATENCY_CONTINUE_GRAPH_Model(MOE_GRAPH_Model):
    def __init__(self, group_ep):
        super().__init__()
        print("Init group_ep")
        self.group_ep = group_ep
        self.distribute_buffer = MoeDistributeBuffer(group_ep, buff_size_lowlatency) if is_lowlatency == 1 else None

    def inner_forward(
        self, x, expert_ids, expert_scales, combine_expert_scales, x_active_mask, group_ep, group_tp, ep_rank_id, tp_rank_id,
        ep_world_size, tp_world_size, expert_shard_type, shared_expert_rank_num, num_experts,
        shared_expert_num, scales, quant_mode, num_max_dispatch_tokens_per_rank, elastic_info,
        moe_expert_num, global_bs, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
        zero_expert_num, copy_expert_num, const_expert_num, dispatch_comm_alg, combine_comm_alg,
        shared_expert_x, golden_expand_x, comm_quant_mode, ori_x, expert_token_nums_type,
        y_dtype=None, x_dtype=None, scales_dtype=None
    ):
        x_combine_res = None
        profiler_expand_x = None
        local_loop = loop if is_performance else 1
        for _ in range(local_loop):
            dispatch_kwargs = {
                'x': x, 'expert_ids': expert_ids, 'expert_scales': expert_scales,
                'x_active_mask': x_active_mask, 'group_ep': group_ep, 'group_tp': group_tp,
                'ep_rank_id': ep_rank_id, 'tp_rank_id': tp_rank_id, 'ep_world_size': ep_world_size,
                'tp_world_size': tp_world_size, 'expert_shard_type': expert_shard_type,
                'shared_expert_num': shared_expert_num, 'shared_expert_rank_num': shared_expert_rank_num,
                'moe_expert_num': moe_expert_num, 'scales': scales, 'quant_mode': quant_mode,
                'global_bs': global_bs, 'elastic_info': elastic_info,
                'expert_token_nums_type': expert_token_nums_type, 'zero_expert_num': zero_expert_num,
                'copy_expert_num': copy_expert_num, 'const_expert_num': const_expert_num,
                'comm_alg': dispatch_comm_alg, 'y_dtype': y_dtype, 'x_dtype': x_dtype,
                'scales_dtype': scales_dtype,
            }
            output_dispatch_npu = call_low_latency_dispatch(self.distribute_buffer, dispatch_kwargs)

            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype not in (torch.float16, torch.bfloat16):
                expand_x_npu = convert_expand_x_dtype(expand_x_npu)
            if is_continue_golden_expand_x:
                expand_x_npu = golden_expand_x
            profiler_expand_x = expand_x_npu

            combine_kwargs = {
                'expand_x': expand_x_npu, 'expert_ids': expert_ids,
                'assist_info_for_combine': assist_info_for_combine_npu,
                'ep_send_counts': ep_recv_counts_npu, 'tp_send_counts': tp_recv_counts_npu,
                'expert_scales': combine_expert_scales, 'x_active_mask': x_active_mask,
                'expand_scales': expand_scales, 'shared_expert_x': shared_expert_x,
                'ori_x': ori_x, 'const_expert_alpha_1': const_expert_alpha_1,
                'const_expert_alpha_2': const_expert_alpha_2, 'const_expert_v': const_expert_v,
                'group_ep': group_ep, 'group_tp': group_tp, 'ep_rank_id': ep_rank_id,
                'tp_rank_id': tp_rank_id, 'ep_world_size': ep_world_size, 'tp_world_size': tp_world_size,
                'expert_shard_type': expert_shard_type, 'shared_expert_num': shared_expert_num,
                'shared_expert_rank_num': shared_expert_rank_num, 'moe_expert_num': moe_expert_num,
                'comm_quant_mode': comm_quant_mode, 'global_bs': global_bs,
                'elastic_info': elastic_info, 'zero_expert_num': zero_expert_num,
                'copy_expert_num': copy_expert_num, 'const_expert_num': const_expert_num,
                'comm_alg': combine_comm_alg,
            }
            output_combine_npu = call_low_latency_combine(self.distribute_buffer, combine_kwargs)
            if x_combine_res is None:
                x_combine_res = output_combine_npu
            x = (
                convert_combine_output_to_input_dtype(output_combine_npu)
                if is_noq_special else output_combine_npu
            )
        return [x_combine_res, output_combine_npu, profiler_expand_x]


# endregion

# region npu执行算子相关函数

def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")


def get_hcomm_info(rank, comm_group):
    if torch.__version__ > '2.0.1':
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info


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

    ep_hcomm_info = get_hcomm_info(rank, ep_group)
    tp_hcomm_info = get_hcomm_info(rank, tp_group)
    print(f"device_{rank} ep_hcomm_info:{ep_hcomm_info} tp_hcomm_info:{tp_hcomm_info}")

    return ep_hcomm_info, tp_hcomm_info, ep_group, tp_group


def make_tp_recv_counts(expert_token_nums):
    real_rank_token_num = expert_token_nums.sum() if expert_token_nums_type == 1 else expert_token_nums[-1]
    return real_rank_token_num.reshape(1).to(torch.int32)


def build_expert_scales_input(expert_scales, mode, device):
    """根据全局模式构造dispatch/combine的expertScale输入。"""
    if mode == 1:
        return None
    if mode == 2:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if expert_scales is None:
        return None
    return expert_scales.to(device=device, dtype=torch.float32)


def convert_to_deep_ep_dispatch_kwargs(dispatch_kwargs):
    return {
        'x': dispatch_kwargs['x'],
        'topk_idx': dispatch_kwargs['expert_ids'],
        'topk_weights': dispatch_kwargs.get('expert_scales'),
        'x_active_mask': dispatch_kwargs['x_active_mask'],
        'num_experts': dispatch_kwargs['moe_expert_num'],
        'quant_mode': dispatch_kwargs['quant_mode'],
        'shared_expert_num': dispatch_kwargs['shared_expert_num'],
        'shared_expert_rank_num': dispatch_kwargs['shared_expert_rank_num'],
        'x_smooth_scale': dispatch_kwargs['scales'],
        'num_max_dispatch_tokens_per_rank': dispatch_kwargs['global_bs'] // dispatch_kwargs['ep_world_size'],
        'elastic_info': dispatch_kwargs['elastic_info'],
        'expert_shard_type': dispatch_kwargs['expert_shard_type'],
        'expert_token_nums_type': dispatch_kwargs['expert_token_nums_type'],
        'zero_expert_num': dispatch_kwargs['zero_expert_num'],
        'copy_expert_num': dispatch_kwargs['copy_expert_num'],
        'const_expert_num': dispatch_kwargs['const_expert_num'],
        'comm_alg': dispatch_kwargs['comm_alg'],
        'y_dtype': dispatch_kwargs.get('y_dtype'),
        'x_dtype': dispatch_kwargs.get('x_dtype'),
        'x_smooth_scales_dtype': dispatch_kwargs.get('scales_dtype'),
    }


def convert_to_deep_ep_combine_kwargs(combine_kwargs):
    return {
        'x': combine_kwargs['expand_x'],
        'topk_idx': combine_kwargs['expert_ids'],
        'assist_info_for_combine': combine_kwargs['assist_info_for_combine'],
        'ep_send_counts': combine_kwargs['ep_send_counts'],
        'topk_weights': combine_kwargs['expert_scales'],
        'x_active_mask': combine_kwargs['x_active_mask'],
        'expand_scales': combine_kwargs.get('expand_scales'),
        'shared_expert_x': combine_kwargs['shared_expert_x'],
        'num_experts': combine_kwargs['moe_expert_num'],
        'expert_shared_type': combine_kwargs['expert_shard_type'],
        'shared_expert_num': combine_kwargs['shared_expert_num'],
        'shared_expert_rank_num': combine_kwargs['shared_expert_rank_num'],
        'comm_alg': combine_kwargs['comm_alg'],
        'comm_quant_mode': combine_kwargs['comm_quant_mode'],
        'num_max_dispatch_tokens_per_rank': combine_kwargs['global_bs'] // combine_kwargs['ep_world_size'],
        'ori_x': combine_kwargs['ori_x'],
        'const_expert_alpha_1': combine_kwargs['const_expert_alpha_1'],
        'const_expert_alpha_2': combine_kwargs['const_expert_alpha_2'],
        'const_expert_v': combine_kwargs['const_expert_v'],
        'elastic_info': combine_kwargs['elastic_info'],
        'zero_expert_num': combine_kwargs['zero_expert_num'],
        'copy_expert_num': combine_kwargs['copy_expert_num'],
        'const_expert_num': combine_kwargs['const_expert_num'],
    }


def call_low_latency_dispatch(ep_buffer, dispatch_kwargs):
    if is_lowlatency == 0:
        torch_kwargs = dict(dispatch_kwargs)
        if not include_tp_info:
            torch_kwargs.pop('group_tp', None)
            torch_kwargs.pop('tp_rank_id', None)
        return torch_npu.npu_moe_distribute_dispatch_v2(**torch_kwargs)

    if ep_buffer is None:
        raise RuntimeError("ep_buffer must be provided for DeepEP low_latency dispatch")
    (
        expand_x, dynamic_scales, expand_idx,
        expert_token_nums, ep_recv_counts, expand_scales
    ) = ep_buffer.low_latency_dispatch(**convert_to_deep_ep_dispatch_kwargs(dispatch_kwargs))
    tp_recv_counts = make_tp_recv_counts(expert_token_nums)
    return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales


def call_low_latency_combine(ep_buffer, combine_kwargs):
    if is_lowlatency == 0:
        torch_kwargs = dict(combine_kwargs)
        if torch_kwargs.get('expert_scales') is None:
            raise ValueError("torch_npu路径的combine不支持expertScales为None，请使用有效Tensor或一维空Tensor")
        if not include_tp_info:
            torch_kwargs.pop('group_tp', None)
            torch_kwargs.pop('tp_rank_id', None)
        return torch_npu.npu_moe_distribute_combine_v2(**torch_kwargs)

    if ep_buffer is None:
        raise RuntimeError("ep_buffer must be provided for DeepEP low_latency combine")
    return ep_buffer.low_latency_combine(**convert_to_deep_ep_combine_kwargs(combine_kwargs))


def warm_up_dispatch(rank, group_ep, group_tp, dispatch_comm_alg="", ep_buffer=None):
    if is_noq_forward:
        x_dtype_mapped = dtype_map.get(x_dtype, "")
        if x_dtype_mapped == "hifloat8":
            from en_dtypes import hifloat8 as hifp8_np_dtype
            x_warm_up_fp = torch.empty(size=[1, h], dtype=torch.float16).uniform_(-1024, 1024)
            x_warm_up_np = x_warm_up_fp.numpy().astype(np.float32).astype(hifp8_np_dtype)
            x_warm_up = torch.from_numpy(x_warm_up_np.view(np.uint8).copy()).npu()
        elif x_dtype_mapped.startswith("float4"):
            x_warm_up = torch.randint(0, 256, size=[1, h // 2], dtype=torch.uint8).npu()
        else:  # float8_e5m2 / float8_e4m3fn
            x_warm_up_fp = torch.empty(size=[1, h], dtype=torch.float16).uniform_(-1024, 1024)
            x_warm_up = x_warm_up_fp.to(input_dtype).npu()
    else:
        x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
    expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()

    dispatch_kwargs_before = get_dispatch_kwargs_warmup(
        x=x_warm_up,
        expert_ids=expert_ids_warm_up,
        group_ep=group_ep,
        group_tp=group_tp,
        ep_rank_id=rank // tp_world_size,
        tp_rank_id=rank % tp_world_size,
        dispatch_comm_alg=dispatch_comm_alg,
    )
    if is_lowlatency == 1 and ep_buffer is None:
        raise RuntimeError("ep_buffer must be provided for DeepEP warmup")
    (
        expand_x, dynamic_scales, expand_idx,
        expert_token_nums, ep_recv_counts, tp_recv_counts, _
    ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs_before)

    return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts


def get_dispatch_x_dtype():
    if is_lowlatency == 0:
        return get_dtype_int_value(x_dtype)
    mapped_dtype = dtype_map.get(x_dtype, "")
    if mapped_dtype in ("hifloat8", "float4_e2m1", "float4_e1m2"):
        return get_dtype_int_value(x_dtype)
    return None


def get_dispatch_kwargs(
        x, expert_ids, x_active_mask, scales,
        group_ep, group_tp, ep_rank_id, tp_rank_id, elastic_info, dispatch_comm_alg,
        expert_scales=None,
):
    x = x.to(input_dtype).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    expert_scales = build_expert_scales_input(expert_scales, dispatch_expertscale, x.device)
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask or is_2d_mask else None
    if elastic_info is not None:
        elastic_info = elastic_info.to(torch.int32).npu() if is_elastic_info else None
    if ((is_quant and is_quant_scales) or is_noq_forward) and scales is not None:
        if is_noq_forward and scales_dtype == "fp8_e8m0":
            scales = scales.npu()  # fp8_e8m0 scales存储为uint8，不能转float32
        else:
            scales = scales.to(torch.float32).npu()

    return {
        'x': x,
        'expert_ids': expert_ids,
        'expert_scales': expert_scales,
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
        'elastic_info': elastic_info,
        'zero_expert_num': zero_expert_num,
        'copy_expert_num': copy_expert_num,
        'const_expert_num': const_expert_num,
        'y_dtype': get_dtype_int_value(expandx_dtype),
        'x_dtype': get_dispatch_x_dtype(),
        'scales_dtype': get_dtype_int_value(scales_dtype) if is_quant_scales else None,
        'expert_token_nums_type': expert_token_nums_type,
        'comm_alg': dispatch_comm_alg
    }


def get_dispatch_kwargs_warmup(
        x, expert_ids, group_ep, group_tp, ep_rank_id, tp_rank_id, dispatch_comm_alg
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
        'elastic_info': None,
        'expert_token_nums_type': expert_token_nums_type,
        'comm_alg': dispatch_comm_alg,
        'expert_scales': None,
        'zero_expert_num': 0,
        'copy_expert_num': 0,
        'const_expert_num': 0,
        'y_dtype': get_dtype_int_value(expandx_dtype),
        'x_dtype': get_dispatch_x_dtype(),
        'scales_dtype': get_dtype_int_value(scales_dtype) if is_quant_scales else None,
    }


def get_combine_kwargs(
        expand_x, expert_ids, expand_idx, ep_send_counts, tp_send_counts, expert_scales,
        x_active_mask, shared_expert_x, group_ep, group_tp, ep_rank_id, tp_rank_id, elastic_info,
        ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
):
    if is_noq_special and expand_x is not None:
        expand_x = convert_expand_x_dtype(expand_x).npu()
    else:
        expand_x = expand_x.to(combine_input_dtype).npu() if expand_x is not None else None
    expert_ids = expert_ids.to(torch.int32).npu()
    expand_idx = expand_idx.to(torch.int32).npu() if expand_x is not None else None
    ep_send_counts = ep_send_counts.to(torch.int32).npu() if ep_send_counts is not None else None
    tp_send_counts = tp_send_counts.to(torch.int32).npu()
    expert_scales = build_expert_scales_input(expert_scales, combine_expertscale, expert_ids.device)
    x_active_mask = x_active_mask.to(torch.bool).npu() if is_mask or is_2d_mask else None
    shared_expert_x = shared_expert_x.to(combine_input_dtype).npu() if is_shared_expert_x else None
    if elastic_info is not None:
        elastic_info = elastic_info.to(torch.int32).npu() if is_elastic_info else None
    # ori_x = ori_x.to(combine_input_dtype).npu() if copy_expert_num or const_expert_num else None
    if copy_expert_num or const_expert_num:
        ori_x = (convert_expand_x_dtype(ori_x) if is_noq_special else ori_x.to(combine_input_dtype)).npu()
    else:
        ori_x = None
    const_expert_alpha_1 = const_expert_alpha_1.to(combine_input_dtype).npu() if const_expert_num else None
    const_expert_alpha_2 = const_expert_alpha_2.to(combine_input_dtype).npu() if const_expert_num else None
    const_expert_v = const_expert_v.to(combine_input_dtype).npu() if const_expert_num else None

    return {
        'expand_x': expand_x,
        'expert_ids': expert_ids,
        'assist_info_for_combine': expand_idx,
        'ep_send_counts': ep_send_counts,
        'tp_send_counts': tp_send_counts,
        'x_active_mask': x_active_mask,
        'expert_scales': expert_scales,
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
        'comm_alg': combine_comm_alg
    }

def apply_device_limit() -> None:
        if not is_full_core:
            torch.npu.set_device_limit(torch.npu.current_device(), aic_core_num, aiv_core_num)
        print(f"当前单算子模式仅使用：{aic_core_num} aic核，{aiv_core_num} aiv核")


def run_dispatch_npu(queue, rank, x, expert_ids, x_active_mask, scales, expert_scales, available_ranks, elastic_info,
                     slow_rank):
    try:
        set_device(rank)
        ep_hcomm_info, tp_hcomm_info, ep_group, _ = init_hccl_comm(rank)
        performance_info = torch.zeros(ep_world_size, dtype=torch.int64).npu()
        if is_save_tensor_to_bin:
            save_tensor_to_bin(x.to(torch.float32), "x_in_" + str(rank))
            save_tensor_to_bin(expert_ids.to(torch.int32), "expertIds_" + str(rank))
            if expert_scales is not None:
                save_tensor_to_bin(expert_scales.to(torch.float32), "expertScales_" + str(rank))
            if ((is_quant and is_quant_scales) or is_noq_forward) and scales is not None:
                save_tensor_to_bin(scales, "scales_" + str(rank))

        print(f'[INFO] device_{rank} 构造dispatch算子输入数据')
        dispatch_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=elastic_info,
            dispatch_comm_alg=dispatch_comm_alg,
            expert_scales=expert_scales,
        )

        dispatch_noelastic_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=None,
            dispatch_comm_alg=dispatch_comm_alg,
            expert_scales=expert_scales,
        )
        if graph_type == 0:
            apply_device_limit()
            ep_buffer = MoeDistributeBuffer(ep_group, buff_size_lowlatency) if is_lowlatency == 1 else None
            if not is_performance:
                if is_elastic:
                    expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                        rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                    torch.npu.synchronize()
                    print(f"====================warm_up_over: rank {rank}=====================")
                    if rank in available_ranks:
                        if is_performance_info:
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            (
                                expand_x, dynamic_scales, expand_idx,
                                expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                            ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                        else:
                            (
                                expand_x, dynamic_scales, expand_idx,
                                expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                            ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                else:
                    if is_performance_info:
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        (
                            expand_x, dynamic_scales, expand_idx,
                            expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                        ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                    else:
                        (
                            expand_x, dynamic_scales, expand_idx,
                            expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                        ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)


            else:
                if is_elastic:
                    expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                        rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                    torch.npu.synchronize()
                    print(f"====================warm_up_over: rank {rank}=====================")
                    if rank in available_ranks:
                        tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                        with get_profiling(loop, repeat, warmup, dispatch_profiling_path) as prof:
                            for i in range(loop):
                                torch.npu.synchronize()
                                (
                                    expand_x, dynamic_scales, expand_idx,
                                    expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                                ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                                if i == 1:
                                    for _ in range(100):
                                        torch.exp(tensor)

                        prof.step()
                        if is_performance_info:
                            dist.barrier()
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            (
                                expand_x, dynamic_scales, expand_idx,
                                expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                            ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                else:
                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(loop, repeat, warmup, dispatch_profiling_path) as prof:
                        for i in range(loop):
                            torch.npu.synchronize()
                            (
                                expand_x, dynamic_scales, expand_idx,
                                expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                            ) = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)

                            prof.step()

        else:
            model = compile_model(MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(ep_group))
            if is_elastic:
                (
                    expand_x, dynamic_scales, expand_idx,
                    expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                ) = model(**dispatch_noelastic_kwargs)
                print(f"====================warm_up_over: rank {rank}=====================")
                if rank in available_ranks:
                    if is_performance_info:
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        (
                            expand_x, dynamic_scales, expand_idx,
                            expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                        ) = model(**dispatch_kwargs)
                    else:
                        (
                            expand_x, dynamic_scales, expand_idx,
                            expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                        ) = model(**dispatch_kwargs)
            else:
                if is_performance_info:
                    if rank in slow_rank:
                        time.sleep(slow_rank_time)
                    (
                        expand_x, dynamic_scales, expand_idx,
                        expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                    ) = model(**dispatch_kwargs)
                else:
                    (
                        expand_x, dynamic_scales, expand_idx,
                        expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales
                    ) = model(**dispatch_kwargs)
        time.sleep(5 if not is_elastic else 20)
        torch.npu.synchronize()
        dump_fullmesh_v2_profiler(expand_x, rank)
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')
        if is_save_tensor_to_bin:
            save_tensor_to_bin(expand_x.to(torch.float32), "expand_x_out_" + str(rank))
            save_tensor_to_bin(expand_idx, "expand_idx_" + str(rank))
            save_tensor_to_bin(expert_token_nums, "expert_token_nums_" + str(rank))
            save_tensor_to_bin(ep_recv_counts, "ep_recv_counts_" + str(rank))
            save_tensor_to_bin(tp_recv_counts, "tp_recv_counts_" + str(rank))
            save_tensor_to_bin(expand_scales, "expand_scales_" + str(rank))
            if is_quant or is_noq_forward:
                save_tensor_to_bin(dynamic_scales, "dynamic_scales_" + str(rank))

        # 获得expand_x, expand_idx和dynamic_scales的真实值，并放入队列
        # 注意, expand_idx无法用相同方式获取真实值，因为real_rank_token_num是all gather之后的token数
        if is_elastic and rank < available_ranks[shared_expert_rank_num_ela]:
            real_rank_token_num = expert_token_nums.cpu()[0].item()
        else:
            real_rank_token_num = expert_token_nums.cpu().sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums.cpu()[-1].item()
        real_origin_token_num = (
            tp_recv_counts[rank % tp_world_size].cpu().item()
            if tp_world_size > 1
            else real_rank_token_num
        )
        expand_x = expand_x[:real_rank_token_num, :]
        if is_ccu:
            expand_idx = expand_idx[:torch.sum(x_active_mask) * k]  # CCU dispatch_v2 用的不是三元组
        else: #MTE
            expand_idx = expand_idx[:real_origin_token_num * 3]
        dynamic_scales = dynamic_scales[:real_rank_token_num]
        expand_scales = expand_scales[:real_rank_token_num]
        print("=========================Queue.put=========================")
        print("child queue.put", flush=True)
        queue.put([
            rank,
            [
                expand_x.cpu(), dynamic_scales.cpu(), expand_idx.cpu(),
                expert_token_nums.cpu(), ep_recv_counts.cpu(), tp_recv_counts.cpu(), expand_scales.cpu()
            ]
        ])
        print("child queue.put end", flush=True)
        print("=========================performance_info=========================")
        print(f"performance_info after Independent npu_moe_distribute_dispatchv2:{performance_info}")
        print(f"dispatchv2 slow_rank:{slow_rank}", flush=True)
    except Exception as e:
        print("=========================Exception catch=========================")
        error_msg = f"Rank {rank} failed with error:\n{traceback.format_exc()}"
        print(error_msg)
        queue.put([rank, None])
        print("=========================Print err=========================")


def run_combine_npu(
        queue, x, scales, rank, expand_x, expert_ids, expand_idx, ep_send_counts, tp_send_counts, x_active_mask,
        expert_scales, shared_expert_x, available_ranks, elastic_info,
        ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, slow_rank,
):
    try:
        set_device(rank)
        ep_hcomm_info, tp_hcomm_info, ep_group, _ = init_hccl_comm(rank)
        performance_info = torch.zeros(ep_world_size, dtype=torch.int64).npu()
        if is_save_tensor_to_bin:
            save_tensor_to_bin(expand_x.to(torch.float32), "expand_x_" + str(rank))
            save_tensor_to_bin(expert_ids.to(torch.int32), "expertIds_" + str(rank))
            save_tensor_to_bin(expand_idx.to(torch.int32), "expandIdx_" + str(rank))
            save_tensor_to_bin(ep_send_counts.to(torch.int32), "epSendCounts_" + str(rank))
            save_tensor_to_bin(tp_send_counts.to(torch.int32), "tpSendCounts_" + str(rank))
            if expert_scales is not None:
                save_tensor_to_bin(expert_scales.to(torch.float32), "expertScales_" + str(rank))

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
            ori_x=ori_x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=elastic_info,
        )
        dispatch_noelastic_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=None,
            dispatch_comm_alg=dispatch_comm_alg,
        )
        if graph_type == 0:
            apply_device_limit()
            ep_buffer = MoeDistributeBuffer(ep_group, buff_size_lowlatency) if is_lowlatency == 1 else None
            if not is_performance:
                if is_elastic:
                    x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                        rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                    torch.npu.synchronize()
                    print(f"====================warm_up_over: rank {rank}=====================")
                if rank in available_ranks:
                    if is_performance_info:
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                        print(f"rank is :{combine_kwargs['ep_rank_id']} expert_ids :{combine_kwargs['expert_ids']}\n")
                        print(f"elastic_info :{combine_kwargs['elastic_info']}\n")
                    else:
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                else:
                    if is_performance_info:
                        x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                            rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                        torch.npu.synchronize()
                        print("====================warm_up_over=====================")
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                        print(f"rank is :{combine_kwargs['ep_rank_id']} expert_ids :{combine_kwargs['expert_ids']}\n")
                        print(f"elastic_info :{combine_kwargs['elastic_info']}\n")
                    else:
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
            else:
                if is_elastic:
                    x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                        rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                    torch.npu.synchronize()
                    print("====================warm_up_over=====================")
                    if rank in available_ranks:
                        tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                        with get_profiling(loop, repeat, warmup, combine_profiling_path) as prof:
                            for i in range(loop):
                                _ = call_low_latency_dispatch(ep_buffer, dispatch_noelastic_kwargs)
                                x = call_low_latency_combine(ep_buffer, combine_kwargs)

                                if i == 1:
                                    for _ in range(100):
                                        torch.exp(tensor)

                                prof.step()

                        if is_performance_info:
                            _ = call_low_latency_dispatch(ep_buffer, dispatch_noelastic_kwargs)
                            dist.barrier()
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            x = call_low_latency_combine(ep_buffer, combine_kwargs)
                else:
                    x = call_low_latency_combine(ep_buffer, combine_kwargs)

                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(loop, repeat, warmup, combine_profiling_path) as prof:
                        for i in range(loop):

                            # 做全卡同步，因为combine只会同步一部分卡
                            dist.all_reduce(combine_kwargs['expand_x'][0, 0], op=ReduceOp.SUM, async_op=False,
                                            group=ep_group)
                            _ = call_low_latency_combine(ep_buffer, combine_kwargs)

                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)

                            prof.step()
                    if is_performance_info:
                        # 做全卡同步，因为combine只会同步一部分卡
                        dist.all_reduce(combine_kwargs['expand_x'][0, 0], op=ReduceOp.SUM, async_op=False,
                                        group=ep_group)
                        dist.barrier()
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        _ = call_low_latency_combine(ep_buffer, combine_kwargs)

        else:
            model = compile_model(MOE_LOW_LATENCY_COMBINE_GRAPH_Model(ep_group))
            if is_elastic:
                warmup_model = compile_model(MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(ep_group))
                (
                    x, dynamic_scales, expand_idx,
                    expert_token_nums, ep_recv_counts, tp_recv_counts, _
                ) = warmup_model(**dispatch_noelastic_kwargs)
                print("====================warm_up_over=====================")
                if rank in available_ranks:
                    if is_performance_info:
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        x = model(**combine_kwargs)
                    else:
                        x = model(**combine_kwargs)
            else:
                if is_performance_info:
                    warmup_model = compile_model(MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(ep_group))
                    (
                        x, dynamic_scales, expand_idx,
                        expert_token_nums, ep_recv_counts, tp_recv_counts, _
                    ) = warmup_model(**dispatch_noelastic_kwargs)
                    if rank in slow_rank:
                        time.sleep(slow_rank_time)
                    x = model(**combine_kwargs)
                    print(f"rank is :{combine_kwargs['ep_rank_id']} expert_ids :{combine_kwargs['expert_ids']}\n")
                else:
                    x = model(**combine_kwargs)
        time.sleep(5 if not is_elastic else 20)
        torch.npu.synchronize()
        dump_fullmesh_v2_profiler(combine_kwargs["expand_x"], rank)
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')
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
        print("=========================performance_info=========================")
        print(f"performance_info after Independent npu_moe_distribute_combinev2:{performance_info}")
        print(f"combinev2 slow_rank:{slow_rank}\n")
    except Exception as e:
        print("=========================Exception catch=========================")
        error_msg = f"Rank {rank} failed with error:\n{traceback.format_exc()}"
        print(error_msg)
        queue.put([rank, None])
        print("=========================Print err=========================")


def run_continue_npu(queue, rank, x, expert_ids, x_active_mask, scales, expert_scales, shared_expert_x, golden_expand_x,
                    available_ranks, elastic_info,
                    ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, slow_rank,):
    try:
        set_device(rank)
        ep_hcomm_info, tp_hcomm_info, ep_group, _ = init_hccl_comm(rank)
        dispatch_performance_info = torch.zeros(ep_world_size, dtype=torch.int64).npu()
        combine_performance_info = torch.zeros(ep_world_size, dtype=torch.int64).npu()
        performance_info = torch.zeros(ep_world_size, dtype=torch.int64).npu() # 图模式下continue需要
        profiler_expand_x = None

        print(f'[INFO] device_{rank} 构造两算子输入数据')
        expert_scales = expert_scales.to(torch.float32).npu() if expert_scales is not None else None  # 图模式下传参需要
        shared_expert_x = shared_expert_x.to(combine_input_dtype).npu() if is_shared_expert_x else None  # 图模式下传参需要
        if is_noq_special:
            # noq_hifp8场景下golden_expand_x是hifp8(uint8)，需要通过convert_expand_x_dtype正确转换
            golden_expand_x = convert_expand_x_dtype(golden_expand_x).npu() if golden_expand_x is not None else None
        else:
            golden_expand_x = golden_expand_x.to(input_dtype).npu() if golden_expand_x is not None else None

        dispatch_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=elastic_info,
            dispatch_comm_alg=dispatch_comm_alg,
            expert_scales=expert_scales,
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
            ori_x=ori_x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=elastic_info,
        )
        dispatch_noelastic_kwargs = get_dispatch_kwargs(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            scales=scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            elastic_info=None,
            dispatch_comm_alg=dispatch_comm_alg,
            expert_scales=expert_scales,
        )
        if graph_type == 0:
            apply_device_limit()
            ep_buffer = MoeDistributeBuffer(ep_group, buff_size_lowlatency) if is_lowlatency == 1 else None
            if not is_performance:
                if is_elastic:
                    x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                        rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                    torch.npu.synchronize()
                    print("====================warm_up_over=====================")
                    if rank in available_ranks:
                        if is_performance_info:
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                        expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                        print("=========================performance_info=========================")
                        print(f"performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")

                        if is_continue_golden_expand_x:
                            expand_x = golden_expand_x
                        else:
                            if is_quant or is_noq_special:
                                expand_x = convert_expand_x_dtype(expand_x)
                            real_rank_token_num = expert_token_nums.sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums[-1].item()
                            expand_x[real_rank_token_num:, :] = 0
                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                            ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                        torch.npu.synchronize()
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)

                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                        print("=========================performance_info=========================")
                        print(f"performance_info after Independent npu_moe_distribute_combinev2:{combine_performance_info}\n")
                    else:
                        expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                        if is_continue_golden_expand_x:
                            expand_x = golden_expand_x
                        else:
                            if is_quant or is_noq_special:
                                expand_x = convert_expand_x_dtype(expand_x)
                            real_rank_token_num = expert_token_nums.sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums[-1].item()
                            expand_x[real_rank_token_num:, :] = 0

                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                            ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                else:
                    if is_performance_info:
                        x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                            rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                        torch.npu.synchronize()
                        print("====================warm_up_over=====================")
                        if rank in slow_rank:
                            print(f"Rank {rank} Sleep {slow_rank_time}")
                            time.sleep(slow_rank_time)

                        expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                        print("=========================performance_info=========================")
                        print(f"rank is :{dispatch_kwargs['ep_rank_id']} ep_recv_counts :{ep_recv_counts}\n")
                        print(f"rank is :{combine_kwargs['ep_rank_id']} expert_ids :{combine_kwargs['expert_ids']}\n")
                        print(f"performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")

                        if is_continue_golden_expand_x:
                            expand_x = golden_expand_x
                        else:
                            if is_quant or is_noq_special:
                                expand_x = convert_expand_x_dtype(expand_x)
                            real_rank_token_num = expert_token_nums.sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums[-1].item()
                            expand_x[real_rank_token_num:, :] = 0

                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                            ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                        torch.npu.synchronize()
                        if rank in slow_rank:
                            print(f"Rank {rank} Sleep {slow_rank_time}")
                            time.sleep(slow_rank_time)
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                        print("=========================performance_info=========================")
                        print(f"rank is :{combine_kwargs['ep_rank_id']} x_activate_mask :{combine_kwargs['x_activate_mask']}\n")
                        print(f"performance_info after Independent npu_moe_distribute_combinev2:{combine_performance_info}\n")
                    else:
                        expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                        torch.npu.synchronize()
                        if is_continue_golden_expand_x:
                            expand_x = golden_expand_x
                        else:
                            if is_quant or is_noq_special:
                                expand_x = convert_expand_x_dtype(expand_x)
                            real_rank_token_num = expert_token_nums.sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums[-1].item()
                            expand_x[real_rank_token_num:, :] = 0
                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                            ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                        torch.npu.synchronize()
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                        torch.npu.synchronize()

            else:
                if is_elastic:
                    x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = warm_up_dispatch(
                        rank, ep_hcomm_info, tp_hcomm_info, dispatch_comm_alg, ep_buffer)
                    torch.npu.synchronize()
                    print("====================warm_up_over=====================")
                    if rank in available_ranks:
                        tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                        with get_profiling(loop, repeat, warmup, continue_profiling_path) as prof:
                            for i in range(loop):

                                expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                                if is_continue_golden_expand_x:
                                    expand_x = golden_expand_x
                                else:
                                    if is_quant or is_noq_special:
                                        expand_x = convert_expand_x_dtype(expand_x)
                                    real_rank_token_num = expert_token_nums.sum().item() \
                                            if expert_token_nums_type == 1 \
                                            else expert_token_nums[-1].item()
                                    expand_x[real_rank_token_num:, :] = 0
                                combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                                    ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)

                                x = call_low_latency_combine(ep_buffer, combine_kwargs)

                                if i == 1:
                                    for _ in range(100):
                                        torch.exp(tensor)

                                prof.step()

                        if is_performance_info:
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                            print("=========================performance_info=========================")
                            print(f"performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")

                            if is_continue_golden_expand_x:
                                expand_x = golden_expand_x
                            else:
                                if is_quant or is_noq_special:
                                    expand_x = convert_expand_x_dtype(expand_x)
                                real_rank_token_num = expert_token_nums.sum().item() \
                                        if expert_token_nums_type == 1 \
                                        else expert_token_nums[-1].item()
                                expand_x[real_rank_token_num:, :] = 0

                            combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                                    ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)

                            x = call_low_latency_combine(ep_buffer, combine_kwargs)

                            expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                            if is_continue_golden_expand_x:
                                expand_x = golden_expand_x
                            else:
                                if is_quant or is_noq_special:
                                    expand_x = convert_expand_x_dtype(expand_x)
                                real_rank_token_num = expert_token_nums.sum().item() \
                                        if expert_token_nums_type == 1 \
                                        else expert_token_nums[-1].item()
                                expand_x[real_rank_token_num:, :] = 0

                            combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                                    ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                            print("====================warm_up_over=====================")
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            x = call_low_latency_combine(ep_buffer, combine_kwargs)
                            print("=========================performance_info=========================")
                            print(f"performance_info after Independent npu_moe_distribute_combinev2:{combine_performance_info}\n")
                else:
                    tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                    with get_profiling(loop, repeat, warmup, continue_profiling_path) as prof:
                        for i in range(loop):

                            expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)

                            if is_continue_golden_expand_x:
                                expand_x = golden_expand_x
                            else:
                                if is_quant or is_noq_special:
                                    expand_x = convert_expand_x_dtype(expand_x)
                                real_rank_token_num = expert_token_nums.sum().item() \
                                        if expert_token_nums_type == 1 \
                                        else expert_token_nums[-1].item()
                                expand_x[real_rank_token_num:, :] = 0
                            combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                                ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)

                            x = call_low_latency_combine(ep_buffer, combine_kwargs)

                            if i == 1:
                                for _ in range(100):
                                    torch.exp(tensor)

                            prof.step()
                    if is_performance_info:
                        dist.barrier()
                        if rank in slow_rank:
                            print(f"Rank {rank} Sleep {slow_rank_time}")
                            time.sleep(slow_rank_time)

                        expand_x, _, expand_idx, _, ep_recv_counts, tp_recv_counts, _ = call_low_latency_dispatch(ep_buffer, dispatch_kwargs)
                        print("=========================performance_info=========================")
                        print(f"performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")
                        if is_continue_golden_expand_x:
                            expand_x = golden_expand_x
                        else:
                            if is_quant or is_noq_special:
                                expand_x = convert_expand_x_dtype(expand_x)
                            real_rank_token_num = expert_token_nums.sum().item() \
                                    if expert_token_nums_type == 1 \
                                    else expert_token_nums[-1].item()
                            expand_x[real_rank_token_num:, :] = 0

                        combine_kwargs.update(expand_x=expand_x, assist_info_for_combine=expand_idx,
                                                ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts)
                        dist.barrier()
                        if rank in slow_rank:
                            print(f"Rank {rank} Sleep {slow_rank_time}")
                            time.sleep(slow_rank_time)
                        x = call_low_latency_combine(ep_buffer, combine_kwargs)
                        print("=========================performance_info=========================")
                        print(f"performance_info after Independent npu_moe_distribute_combinev2:{combine_performance_info}\n")

        else:
            continue_graph_kwargs = {
                **dispatch_kwargs,
                "combine_expert_scales": combine_kwargs["expert_scales"],
                "shared_expert_x": shared_expert_x,
                "golden_expand_x": golden_expand_x,
                'comm_quant_mode': comm_quant_mode,
                'ori_x': combine_kwargs["ori_x"],
                'const_expert_alpha_1': combine_kwargs["const_expert_alpha_1"],
                'const_expert_alpha_2': combine_kwargs["const_expert_alpha_2"],
                'const_expert_v': combine_kwargs["const_expert_v"],
                'num_experts': dispatch_kwargs['moe_expert_num'],
                'num_max_dispatch_tokens_per_rank': dispatch_kwargs['global_bs'] // dispatch_kwargs['ep_world_size'],
            }
            del continue_graph_kwargs["comm_alg"]
            continue_graph_kwargs["dispatch_comm_alg"] = dispatch_comm_alg
            continue_graph_kwargs["combine_comm_alg"] = combine_comm_alg

            if not is_performance:
                if is_elastic:
                    model = compile_model(MOE_LOW_LATENCY_CONTINUE_GRAPH_Model(ep_group))
                    warmup_model = compile_model(MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(ep_group))
                    (
                        x, dynamic_scales, expand_idx,
                        expert_token_nums, ep_recv_counts, tp_recv_counts, _
                    ) = warmup_model(**dispatch_noelastic_kwargs)
                    print("====================warm_up_over=====================")
                    if rank in available_ranks:
                        if is_performance_info:
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            x, _, profiler_expand_x = model(**continue_graph_kwargs)
                            print("=========================performance_info=========================")
                            print(f"continue_performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")
                        else:
                            x, _, profiler_expand_x = model(**continue_graph_kwargs)
                else:
                    if is_performance_info:
                        model = compile_model(MOE_LOW_LATENCY_CONTINUE_GRAPH_Model(ep_group))
                        warmup_model = compile_model(MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(ep_group))
                        (
                            x, dynamic_scales, expand_idx,
                            expert_token_nums, ep_recv_counts, tp_recv_counts, _
                        ) = warmup_model(**dispatch_noelastic_kwargs)
                        print("====================warm_up_over=====================")
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        x, _, profiler_expand_x = model(**continue_graph_kwargs)
                        print("=========================performance_info=========================")
                        print(f"continue_performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")
                    else:
                        model = compile_model(MOE_LOW_LATENCY_CONTINUE_GRAPH_Model(ep_group))
                        x, _, profiler_expand_x = model(**continue_graph_kwargs)

            else:
                model = compile_model(MOE_LOW_LATENCY_CONTINUE_GRAPH_Model(ep_group))
                if is_elastic:
                    warmup_model = compile_model(MOE_LOW_LATENCY_DISPATCH_GRAPH_Model(ep_group))
                    (
                        x, dynamic_scales, expand_idx,
                        expert_token_nums, ep_recv_counts, tp_recv_counts, _
                    ) = warmup_model(**dispatch_noelastic_kwargs)
                    print("====================warm_up_over=====================")
                    if rank in available_ranks:
                        loop_time = 2
                        repeat_time = 1
                        warm_time = 1
                        prof = get_profiling(loop_time, repeat_time, warm_time, continue_profiling_path)
                        prof.start()

                        for i in range(loop_time):
                            x, _, profiler_expand_x = model(**continue_graph_kwargs)
                            prof.step()

                        prof.stop()
                        if is_performance_info:
                            if rank in slow_rank:
                                time.sleep(slow_rank_time)
                            x, _, profiler_expand_x = model(**continue_graph_kwargs)
                            print("=========================performance_info=========================")
                            print(f"continue_performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")

                else:
                    loop_time = 2
                    repeat_time = 1
                    warm_time = 1
                    prof = get_profiling(loop_time, repeat_time, warm_time, continue_profiling_path)
                    prof.start()

                    for i in range(loop_time):
                        x, _, profiler_expand_x = model(**continue_graph_kwargs)
                        prof.step()

                    prof.stop()
                    if is_performance_info:
                        if rank in slow_rank:
                            time.sleep(slow_rank_time)
                        x, _, profiler_expand_x = model(**continue_graph_kwargs)
                        print("=========================performance_info=========================")
                        print(f"continue_performance_info after Independent npu_moe_distribute_dispatchv2:{dispatch_performance_info}\n")

        if graph_type == 0:
            profiler_expand_x = combine_kwargs["expand_x"]
        time.sleep(20)
        torch.npu.synchronize()
        if profiler_expand_x is not None:
            dump_fullmesh_v2_profiler(profiler_expand_x, rank)
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

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
        print(f"continue slow_rank:{slow_rank}\n")
    except Exception as e:
        print("=========================Exception catch=========================")
        error_msg = f"Rank {rank} failed with error:\n{traceback.format_exc()}"
        print(error_msg)
        queue.put([rank, None])
        print("=========================Print err=========================")

def gen_npu(target_func, available_ranks, **server_kwargs):
    def parse_rank_input(target_func, result_queue, rank, server_kwargs):
        def pad_expand_x(expand_x: torch.Tensor, ep_id: int):
            if expand_x is None:
                return
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_x = torch.zeros(size=[tp_world_size * A - expand_x.size(0), h], dtype=expand_x.dtype)
            return torch.cat([expand_x, pad_for_expand_x])

        def pad_expand_idx(expand_idx: torch.Tensor, ep_id: int):
            if expand_x is None:
                return
            if is_ccu:
                return expand_idx
            A = A_shared if ep_id < shared_expert_rank_num else A_moe
            pad_for_expand_idx = torch.zeros(size=[A * 128 - expand_idx.size(0), ], dtype=expand_idx.dtype)
            return torch.cat([expand_idx, pad_for_expand_idx])

        def pad_shared_expert_x(shared_expert_x: torch.Tensor):
            if (not is_shared_expert_x) or (not is_mask):
                return shared_expert_x

            pad_for_shared_expert_x = torch.zeros(size=[bs - shared_expert_x.size(0), h], dtype=shared_expert_x.dtype)
            return torch.cat([shared_expert_x, pad_for_shared_expert_x])

        def filter_tensor(tensor: torch.Tensor, mask: torch.Tensor):
            return bool_index_tensor(tensor, mask) if is_variable_bs else tensor

        def filter_optional_tensor(tensor: torch.Tensor, mask: torch.Tensor):
            return None if tensor is None else filter_tensor(tensor, mask)

        ep_id = rank // tp_world_size
        tp_id = rank % tp_world_size

        mask = server_kwargs["x_active_mask_list"][tp_id][ep_id]

        if target_func == run_dispatch_npu:
            dispatch_scales = server_kwargs["scales_list"][tp_id]
            if is_noq_forward and dispatch_scales is not None:
                # per-token scales，按ep_id切分后再过filter_tensor以匹配real_bs
                dispatch_scales = filter_tensor(dispatch_scales[ep_id * bs:(ep_id + 1) * bs], mask)
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][tp_id][ep_id], mask),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][tp_id][ep_id], mask),
                "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
                "scales": dispatch_scales,
                "expert_scales": filter_tensor(server_kwargs["expert_scales_list"][tp_id][ep_id], mask),
                "available_ranks": available_ranks,
                "elastic_info": server_kwargs['elastic_info'],
                "slow_rank": server_kwargs['slow_rank'],
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
                "expert_scales": filter_optional_tensor(
                    None if server_kwargs["expert_scales_list"] is None
                    else server_kwargs["expert_scales_list"][tp_id][ep_id],
                    mask,
                ),
                "shared_expert_x": pad_shared_expert_x(server_kwargs["shared_expert_x_list"][tp_id][ep_id]),
                "available_ranks": available_ranks,
                "elastic_info": server_kwargs['elastic_info'],
                "ori_x": filter_tensor(server_kwargs["ori_x_list"][tp_id][ep_id], mask),
                "const_expert_alpha_1": server_kwargs["const_expert_alpha_1"],
                "const_expert_alpha_2": server_kwargs["const_expert_alpha_2"],
                "const_expert_v": server_kwargs["const_expert_v"],
                "slow_rank": server_kwargs['slow_rank'],
            }

        if target_func == run_continue_npu:
            continue_scales = server_kwargs["scales_list"][tp_id]
            if is_noq_forward and continue_scales is not None:
                # per-token scales，按ep_id切分后再过filter_tensor以匹配real_bs
                continue_scales = filter_tensor(continue_scales[ep_id * bs:(ep_id + 1) * bs], mask)
            return {
                "queue": result_queue,
                "rank": rank,
                "x": filter_tensor(server_kwargs["x_list"][tp_id][ep_id], mask),
                "expert_ids": filter_tensor(server_kwargs["expert_ids_list"][tp_id][ep_id], mask),
                "x_active_mask": server_kwargs["x_active_mask_list"][tp_id][ep_id],
                "scales": continue_scales,
                "expert_scales": filter_optional_tensor(
                    None if server_kwargs["expert_scales_list"] is None
                    else server_kwargs["expert_scales_list"][tp_id][ep_id],
                    mask,
                ),
                "shared_expert_x": pad_shared_expert_x(server_kwargs["shared_expert_x_list"][tp_id][ep_id]),
                "golden_expand_x": pad_expand_x(server_kwargs["golden_expand_x_list"][tp_id][ep_id], ep_id),
                "available_ranks": available_ranks,
                "elastic_info": server_kwargs['elastic_info'],
                "ori_x": filter_tensor(server_kwargs["ori_x_list"][tp_id][ep_id], mask),
                "const_expert_alpha_1": server_kwargs["const_expert_alpha_1"],
                "const_expert_alpha_2": server_kwargs["const_expert_alpha_2"],
                "const_expert_v": server_kwargs["const_expert_v"],
                "slow_rank": server_kwargs['slow_rank'],
            }

    if target_func not in [run_dispatch_npu, run_combine_npu, run_continue_npu]:
        print("target_func not supported!")
        exit(1)

    if int(server_num) > 1:
        print(f"multi_server scene!!!!! {server_num=}, {world_size=}, {server_index=}. only support running case one by one!!!!!!!!!!!!!")
        rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    else:
        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
    print(f"rank list is: {rank_list}")

    # 多进程，每个进程调用一张卡执行目标函数
    proc_list = []
    mp.set_start_method("forkserver", force=True)
    with Manager() as manager:
        result_queue = manager.Queue()

        for rank in rank_list:
            rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
            proc = Process(target=target_func, kwargs=rank_kwargs)
            print("==============================Start Proc==============================")
            proc.start()
            proc_list.append(proc)

        rank_outputs = [None] * rank_per_dev
        for proc in proc_list:
            print("==============================Begin To Get Proc==============================")
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output
            print("==============================Out To Proc==============================")

        print(f"remaining_data: {result_queue.qsize()}")
        for proc in proc_list:
            proc.join()

    if None in rank_outputs:
        print("[ERROR] Task failed! Please check the detailed error log printed by subprocesses.")
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

def gen_avalible_list():
    shared_list = list(range(shared_expert_rank_num))
    moe_list = list(range(shared_expert_rank_num, world_size))
    all_list = list(range(world_size))
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
            avalible_list = random.sample(range(world_size), world_size - elastic_random_ranks)
    else:
        random.seed(random_seed)
        server_elastic_index = random.randint(0, server_num)
        original_list = list(range(server_num * rank_per_dev))
        start = server_elastic_index * rank_per_dev
        end = start + rank_per_dev
        avalible_list = [x for x in original_list if not (start <= x < end)]
    print(f'{avalible_list=}')
    return sorted(avalible_list) if is_elastic else list(range(world_size)), server_elastic_index


def gen_elastic_info(avalible_list):
    elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype=torch.int32)  # 初始化张量
    # 设置前四个值
    elastic_info[0] = is_elastic  # 是否缩容 (1表示是)
    elastic_info[1] = len(avalible_list) // tp_world_size # 缩容后实际rank数
    elastic_info[2] = shared_expert_rank_num_ela  # 共享专家使用rank数
    elastic_info[3] = (len(avalible_list) // tp_world_size - shared_expert_rank_num_ela) * local_moe_expert_num  # moe专家个数
    table1 = [-1] * ep_world_size  # Table1[epRankId] = localEpRankId或-1
    table2 = [-1] * ep_world_size  # Table2[localEpRankId] = epRankId
    for local_rank_id, ep_rank_id in enumerate(avalible_list):
        if ep_rank_id < ep_world_size:  # 确保ep_rank_id在有效范围内
            table1[ep_rank_id] = local_rank_id  # local_rank_id表示缩容后的ep_rank_id
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
    x_mapped_dtype = dtype_map.get(x_dtype, "")
    for _ in range(tp_world_size):
        if x_mapped_dtype == "hifloat8":
            # hifp8不能直接用uniform_生成，先生成fp16再cast为hifp8
            from en_dtypes import hifloat8 as hifp8_np_dtype
            if is_x_random:
                cur_x_fp = torch.empty(size=[global_bs, h], dtype=torch.float16).uniform_(-5, 5)
            else:
                cur_x_fp = torch.empty(size=[global_bs, h], dtype=torch.float16).fill_(1)
            cur_x_np = cur_x_fp.numpy().astype(np.float32).astype(hifp8_np_dtype)
            cur_x = torch.from_numpy(cur_x_np.view(np.uint8).copy())
        elif x_mapped_dtype.startswith("float4"):
            # fp4: 2个元素packed成1个uint8，shape为[global_bs, h//2]
            if is_x_random:
                cur_x = torch.randint(0, 256, size=[global_bs, h // 2], dtype=torch.uint8)
            else:
                cur_x = torch.zeros(size=[global_bs, h // 2], dtype=torch.uint8)
        elif x_mapped_dtype in ("float8_e5m2", "float8_e4m3fn"):
            # fp8: torch原生支持，但不支持uniform_，先生成fp16再cast
            if is_x_random:
                cur_x_fp = torch.empty(size=[global_bs, h], dtype=torch.float16).uniform_(-5, 5)
            else:
                cur_x_fp = torch.arange(global_bs, dtype=torch.float16).view(-1, 1).repeat(1, h)
            cur_x = cur_x_fp.to(input_dtype)
        else:
            if is_x_random:
                cur_x = torch.empty(size=[global_bs, h], dtype=input_dtype).uniform_(-5, 5)
            else:
                cur_x = torch.arange(global_bs, dtype=input_dtype).view(-1, 1).repeat(1, h)

        x_list.append(cur_x)

    return x_list


def gen_expert_ids(moe_expert_num_re):
    moe_expert_num_re = int(moe_expert_num_re)
    total_expert_num = moe_expert_num_re + zero_expert_num + copy_expert_num + const_expert_num

    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]

    def build_moe_order():
        # 普通 moe 专家按目标 moe 卡轮询排列。
        moe_rank_num_re = (moe_expert_num_re + local_moe_expert_num - 1) // local_moe_expert_num
        experts_by_rank = [
            list(range(
                rank_id * local_moe_expert_num,
                min((rank_id + 1) * local_moe_expert_num, moe_expert_num_re),
            ))
            for rank_id in range(moe_rank_num_re)
        ]

        moe_order = []
        max_local_expert_num = max((len(experts) for experts in experts_by_rank), default=0)
        for local_expert_id in range(max_local_expert_num):
            for experts in experts_by_rank:
                if local_expert_id < len(experts):
                    moe_order.append(experts[local_expert_id])
        return moe_order

    def interleave_specials(moe_order):
        # 特殊专家均匀插入普通 moe 专家序列。
        special_order = list(range(moe_expert_num_re, total_expert_num))
        if not special_order:
            return torch.tensor(moe_order, dtype=torch.int32)

        result = []
        moe_index = 0
        special_index = 0
        special_expert_num = len(special_order)
        for position in range(total_expert_num):
            expected_special_num = (position + 1) * special_expert_num // total_expert_num
            if special_index < expected_special_num:
                result.append(special_order[special_index])
                special_index += 1
            else:
                result.append(moe_order[moe_index])
                moe_index += 1
        return torch.tensor(result, dtype=torch.int32)

    def gen_balanced_expert_ids_per_card(base_order, card_id):
        # 每张源卡取 base_order 上长度为 bs*k 的连续循环片段。
        # base_order 是 [0, total_expert_num) 的全排列，k <= total_expert_num 时单 token 内不会重复。
        start = card_id * bs * k
        expert_indices = (torch.arange(bs * k, dtype=torch.long) + start) % total_expert_num
        return base_order[expert_indices].view(bs, k)

    ep_expert_ids_list = []
    base_order = interleave_specials(build_moe_order()) if is_topk_balance else None
    for _ in range(tp_world_size):
        if is_topk_balance:
            ep_expert_ids = [
                gen_balanced_expert_ids_per_card(base_order, ep_id)
                for ep_id in range(ep_world_size)
            ]
            ep_expert_ids_list.append(torch.cat(ep_expert_ids, dim=0))
        else:
            ep_expert_ids = [gen_random_token_expert_ids(k, 0,
                                                         total_expert_num) \
                             for _ in range(global_bs)]
            ep_expert_ids_list.append(torch.cat(ep_expert_ids).view([global_bs, k]))

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
                scales = torch.empty(size=[1,1], dtype=torch.float32).uniform_(-1, 1)
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
                    actual_shared_expert_num = 0 \
                                                if shared_expert_num == 0 or shared_expert_rank_num == 0 \
                                                else shared_expert_num
                    scales = torch.empty(size=[actual_shared_expert_num + moe_expert_num_list[tp_id], h], dtype=torch.float32).uniform_(-1, 1)
                    padded_scales = torch.cat((
                        scales[:actual_shared_expert_num].repeat_interleave(repeats=rank_num_per_shared_expert, dim=0),scales[actual_shared_expert_num:]
                    ))

        elif is_noq_forward:
            # 场景2：per-token scales
            if dtype_map.get(expandx_dtype, "").startswith("float4"):
                # fp4场景使用fp8_e8m0 scales（存储为uint8，表示共享指数）
                # scales = torch.randint(0, 256, size=[global_bs, scales_h], dtype=torch.uint8)
                scales = torch.empty(size=[global_bs, scales_h], dtype=torch.uint8).fill_(1)
            else:
                # 其他noq forward场景使用float32 scales
                scales = torch.empty(size=[global_bs, scales_h], dtype=torch.float32).uniform_(-1, 1)
            padded_scales = scales  # per-token无需padding

        else:
            # 使用空张量torch.empty(0)会被dispatch算子的tiling检查，只能用None，无法保证数据类型一致
            scales = None
            padded_scales = None
        scales_list.append(scales)
        padded_scales_list.append(padded_scales)

    return scales_list, padded_scales_list


def gen_expert_scales(fill_value=None):
    if fill_value is not None:
        return [
            torch.full(size=[global_bs, k], fill_value=fill_value, dtype=torch.float32)
            for _ in range(tp_world_size)
        ]
    return [torch.empty(size=[global_bs, k], dtype=torch.float32).uniform_(-1, 1) for _ in range(tp_world_size)]


def gen_shared_expert_x(x_active_mask_list: List[List[torch.Tensor]], from_=0, to=1):
    if not is_shared_expert_x:
        return [[torch.empty(0) for _ in range(ep_world_size)] for _ in range(tp_world_size)]

    x_active_mask_per_card = [chunk_tensor(mask_ep, ep_world_size) for mask_ep in x_active_mask_list]

    shared_expert_x_list = []
    for tp_id in range(tp_world_size):
        per_tp = []
        for ep_id in range(ep_world_size):
            size = [bs, h] if is_2d_mask else [x_active_mask_per_card[tp_id][ep_id].sum().item(), h]
            per_tp.append(torch.empty(size=size, dtype=combine_input_dtype).uniform_(from_, to))
        shared_expert_x_list.append(per_tp)

    return shared_expert_x_list


def gen_const_expert_alpha_1():
    return torch.empty(size=[const_expert_num, h], dtype=combine_input_dtype).uniform_(-1, 1)


def gen_const_expert_alpha_2():
    return torch.empty(size=[const_expert_num, h], dtype=combine_input_dtype).uniform_(-1, 1)


def gen_const_expert_v():
    return torch.empty(size=[const_expert_num, h], dtype=combine_input_dtype).uniform_(-1, 1)

def generate_slow_rank(random_slow_rank_num, ep_world_size, specified_slow_rank=None):
    if specified_slow_rank is None:
        # 生成一个包含0到ep_world_size-1的列表
        all_ranks = list(range(ep_world_size))
        # 随机选择random_slow_rank_num个不重复的元素
        slow_rank = random.sample(all_ranks, random_slow_rank_num)
        # 按升序排列
        slow_rank.sort()
    else:
        # 直接使用指定的slow_rank
        slow_rank = specified_slow_rank
        # 确保指定的slow_rank中的元素都在0到ep_world_size-1范围
        if not all(0 <= rank < ep_world_size for rank in slow_rank):
            raise ValueError("specified_slow_rank contains invalid ranks. All ranks must be in range 0 to ep_world_size - 1.")
    return slow_rank


def apply_mapping(tensor_list, map_list):
    mapped_tensors = []
    for t in tensor_list:
        if t is None:
            mapped_tensors.append(None)
            continue
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
        x_list, expert_ids_list, expert_scales_list, x_active_mask_list, x_active_mask_list_ela_ori,
        padded_scales_list, table1_elastic, available_ranks, table2_elastic, elastic_info
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
            x = bool_index_tensor(x, x_active_mask)
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
        return bool_index_tensor(expanded_x, expanded_mask) if is_2d_mask else expanded_x, sorted_col_idx[
            expanded_mask] if is_2d_mask else sorted_col_idx, golden_moe_token

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
                bool_index_tensor(local_x, local_mask.any(dim=1))  # 只保留 mask 至少有一个 True 的行
                for local_x, local_mask in zip(
                    chunk_tensor(x, ep_world_size),
                    chunk_tensor(x_active_mask, ep_world_size)
                )
            ]

        else:
            real_x_per_card = [
                bool_index_tensor(local_x, local_mask)
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
            for shared_card_id in range(shared_expert_rank_num_ela):
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert_ela
                send_card_ids_elastic = list(
                    range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert_ela))
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
        QUANT_MODE_DTYPE = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "int8": "int8",
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "hif8": "hifloat8",
            "fp4_e2m1": "float4_e2m1",
            "fp4_e1m2": "float4_e1m2"
        }
        def cal_quant(y: torch.Tensor, scales: torch.Tensor):
            from quantize import static_quantize, dynamic_quantize, mx_quantize, QUANT_PER_TOKEN
            target_type = QUANT_MODE_DTYPE[expandx_dtype]
            y = torch_to_numpy_tensor(y)
            scales = torch_to_numpy_tensor(scales)
            if quant_mode == 1:
                y = static_quantize(y, scales, dst_type_str=target_type)
                dynamic_scales = np.repeat(scales, (y.shape[0],), axis=0)
                dynamic_scales = dynamic_scales.reshape(y.shape[0], -1)
            if quant_mode == 2:
                y, dynamic_scales = dynamic_quantize(y, dst_type_str=target_type, smooth_scales=scales, quant_mode=QUANT_PER_TOKEN)
            elif quant_mode == 4:
                tile_size = 32
                dynamic_scales, y = mx_quantize(y, target_type, axis=-1, block_size=tile_size, round_mode="rint")
            elif quant_mode == 5:
                tile_size = 32
                dynamic_scales, y = mx_quantize(
                    y, target_type, axis=-1, block_size=tile_size, round_mode="rint", scale_alg=1
                )
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

        # 每个专家的dynamic_scales，一个token一个scales
        shared_dynamic_scales = [torch.empty(0) for _ in range(len(shared_list))]
        moe_dynamic_scales = [torch.empty(0) for _ in range(len(moe_x_segments))]

        for i in range(shared_expert_rank_num):
            # shared_dynamic_scales[i] 为长度为 shared_list[i] 行数的一维张量，长度即每张共享专家卡上的token数
            shared_list[i], shared_dynamic_scales[i] = cal_quant(shared_list[i], scales[i, :] if is_quant_scales else torch.ones((1,)))

        ### 对每个专家的moe_x进行量化（列表moe_x_segments中元素为各moe专家的expand_x）
        _is_fp4_quant = (quant_mode in (4, 5) and QUANT_MODE_DTYPE[expandx_dtype].startswith("float4"))
        for moe_expert_id in range(len(moe_x_segments)):
            # moe_dynamic_scales[i] 长度为 golden_moe_token[i] 大小，即每个moe专家接收到的token数
            if moe_x_segments[moe_expert_id].size(0) > 0:
                moe_x_segments[moe_expert_id], moe_dynamic_scales[moe_expert_id] = cal_quant(
                    moe_x_segments[moe_expert_id],
                    scales[shared_expert_rank_num + moe_expert_id, :] if is_quant_scales else torch.ones((1,))
                )
            elif _is_fp4_quant:
                moe_x_segments[moe_expert_id] = torch.empty((0, h // 2), dtype=torch.uint8)

        # 得到各moe卡的dynamic scales
        rank_expert_id_bounds = torch.cat(
            [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]  # 每个数字, 既是前一张卡的尾，又是后一张卡的头
        ).tolist()
        moe_dynamic_scales_per_card = [
            torch.cat(moe_dynamic_scales[rank_expert_id_bounds[moe_rank_id]: rank_expert_id_bounds[moe_rank_id + 1]])
            for moe_rank_id in range(moe_rank_num)
        ]

        golden_dynamic_scales = shared_dynamic_scales + moe_dynamic_scales_per_card

        return shared_list, moe_x_segments, golden_dynamic_scales

    def gen_triplets_expand_idx(
            ep_local_moe_expert_num: torch.Tensor,
            expert_ids: torch.Tensor, x_active_mask: torch.Tensor, x_active_mask_ela_ori, available_ranks,
            table2_elastic
    ):
        """ 输出expand_x中各token对应的(rank_id, token_id, k_id)"""

        def gen_shared_expand_idx(x_active_mask: torch.Tensor, available_ranks, table2_elastic):
            shared_expand_idx = [[] for _ in range(shared_expert_rank_num)]  # 存储每个共享专家收到的(r,t,k)
            if is_2d_mask:
                real_token_num_list = [bs for i in range(ep_world_size)]
                x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            else:
                real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

            for shared_card_id in range(shared_expert_rank_num_ela):
                shared_expert_index = shared_card_id // rank_num_per_shared_expert_ela
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert_ela
                shared_idx_range = list(
                    range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert_ela)) \
                    if is_elastic else list(range(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert_ela))
                for send_card_id in shared_idx_range:
                    if not is_2d_mask:
                        rtk_list = [
                            ele
                            for token_id in
                            range(real_token_num_list[send_card_id if not is_elastic else table2_elastic[send_card_id]])
                            for ele in (send_card_id if not is_elastic else table2_elastic[send_card_id], token_id,
                                        k + shared_expert_index)
                        ]
                        shared_expand_idx[shared_card_id].extend(rtk_list)
                    else:

                        for token_id in range(bs):
                            if x_active_mask_per_card[send_card_id if not is_elastic else table2_elastic[send_card_id]][
                                token_id].any():
                                rtk_list = [send_card_id if not is_elastic else table2_elastic[send_card_id], token_id,
                                            k + shared_expert_index]

                                shared_expand_idx[shared_card_id].extend(rtk_list)

            return shared_expand_idx

        def gen_moe_expand_idx(
                ep_local_moe_expert_num: torch.Tensor,
                expert_ids: torch.Tensor, x_active_mask: torch.Tensor
        ):
            moe_expand_idx = [[] for _ in range(moe_expert_num)]  # 存储每个moe专家收到的(r,t,k)

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
                [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]  # 每个数字, 既是前一张卡的尾，又是后一张卡的头
            ).tolist()
            moe_expand_idx = [
                sum(moe_expand_idx[rank_expert_id_bounds[moe_rank_id]: rank_expert_id_bounds[moe_rank_id + 1]], [])
                for moe_rank_id in range(moe_rank_num)
            ]

            return moe_expand_idx

        expand_idx = (
                gen_shared_expand_idx(x_active_mask_ela_ori, available_ranks, table2_elastic)
                + gen_moe_expand_idx(ep_local_moe_expert_num, expert_ids, x_active_mask)
        )
        expand_idx = [torch.tensor(local_expand_idx, dtype=torch.int32) for local_expand_idx in expand_idx]  # 格式转换，以便后续处理
        return expand_idx

    def gen_expand_scales(
            ep_local_moe_expert_num: torch.Tensor,
            expert_ids: torch.Tensor, expert_scales: torch.Tensor, x_active_mask: torch.Tensor,
            x_active_mask_ela_ori, available_ranks, table2_elastic
    ):
        def gen_shared_expand_scales(x_active_mask: torch.Tensor, available_ranks, table2_elastic):
            shared_expand_scales = [[] for _ in range(shared_expert_rank_num)]
            if is_2d_mask:
                x_active_mask_per_card = x_active_mask.chunk(ep_world_size)
            else:
                real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

            for shared_card_id in range(shared_expert_rank_num_ela):
                shared_expert_index = shared_card_id // rank_num_per_shared_expert_ela
                card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert_ela
                shared_idx_range = list(
                    range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert_ela)
                ) if is_elastic else list(range(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert_ela))
                for send_card_id in shared_idx_range:
                    original_send_card_id = send_card_id if not is_elastic else table2_elastic[send_card_id]
                    if not is_2d_mask:
                        shared_expand_scales[shared_card_id].extend(
                            [0.0] * real_token_num_list[original_send_card_id]
                        )
                    else:
                        for token_id in range(bs):
                            if x_active_mask_per_card[original_send_card_id][token_id].any():
                                shared_expand_scales[shared_card_id].append(0.0)
            return shared_expand_scales

        def gen_moe_expand_scales(
                ep_local_moe_expert_num: torch.Tensor,
                expert_ids: torch.Tensor, expert_scales: torch.Tensor, x_active_mask: torch.Tensor
        ):
            moe_expand_scales = [[] for _ in range(moe_expert_num)]
            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)
            expert_ids_segments = expert_ids.chunk(ep_world_size)
            expert_scales_segments = expert_scales.chunk(ep_world_size)

            for send_rank_id in range(ep_world_size):
                segment = expert_ids_segments[send_rank_id]
                scales_segment = expert_scales_segments[send_rank_id]
                for token_id in range(bs):
                    if is_2d_mask:
                        for k_id in range(k):
                            if x_active_mask_per_card[send_rank_id][token_id][k_id]:
                                received_moe_expert = segment[token_id, k_id].item()
                                moe_expand_scales[received_moe_expert].append(
                                    scales_segment[token_id, k_id].item()
                                )
                    else:
                        if x_active_mask_per_card[send_rank_id][token_id]:
                            for k_id in range(k):
                                received_moe_expert = segment[token_id, k_id].item()
                                moe_expand_scales[received_moe_expert].append(
                                    scales_segment[token_id, k_id].item()
                                )

            rank_expert_id_bounds = torch.cat(
                [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]
            ).tolist()
            return [
                sum(
                    moe_expand_scales[rank_expert_id_bounds[moe_rank_id]: rank_expert_id_bounds[moe_rank_id + 1]],
                    [],
                )
                for moe_rank_id in range(moe_rank_num)
            ]

        expand_scales = (
            gen_shared_expand_scales(x_active_mask_ela_ori, available_ranks, table2_elastic)
            + gen_moe_expand_scales(ep_local_moe_expert_num, expert_ids, expert_scales, x_active_mask)
        )
        return [torch.tensor(local_expand_scales, dtype=torch.float32) for local_expand_scales in expand_scales]

    def gen_expand_idx(expert_ids: torch.Tensor, x_active_mask: torch.Tensor):
        moe_expand_idx_list = []
        moe_expand_idx_masked_list = []
        expert_ids_segments = expert_ids.chunk(ep_world_size)
        x_active_mask_segments = x_active_mask.chunk(ep_world_size)
        for send_rank_id in range(ep_world_size):
            # 获取本卡发出的expert_ids
            if is_ccu:
                segment = expert_ids_segments[send_rank_id][x_active_mask_segments[send_rank_id]]
            else:
                segment = expert_ids_segments[send_rank_id]
            # 获取各个专家接收到token数量的前缀和
            cumsum_pad = torch.cat([torch.zeros(1, dtype=segment.dtype),
                                    torch.cumsum(torch.bincount(segment.flatten(), minlength=moe_expert_num), dim=0)])
            moe_expand_idx = torch.zeros((bs, k), dtype=torch.int32)
            indices = torch.arange(cumsum_pad.size(0) - 1)
            rank_based_indices = (indices // local_moe_expert_num) * local_moe_expert_num
            # expert_cnts 内的初始值表示本专家在rank内的基础偏移
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
            expert_ids: torch.Tensor, x_active_mask: torch.Tensor, x_active_mask_ela_ori, table1_elastic,
            available_ranks, table2_elastic, elastic_info
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
                for shared_card_id in range(shared_expert_rank_num_ela):
                    card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert_ela
                    send_card_ids_elastic = list(
                        range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert_ela))
                    send_card_ids = [table2_elastic[i] for i in send_card_ids_elastic]
                    shared_expert_recv_count[shared_card_id][send_card_ids_elastic] = real_token_num_tensor[
                        send_card_ids].to(torch.int32)

            # 对每张共享专家卡上的recv_count求累加和，确定发回给每张卡的数据偏移和数据量
            shared_expert_recv_count = shared_expert_recv_count.cumsum(dim=1, dtype=torch.int32)

            shared_expert_recv_count_per_card = [local_ep_recv_count for local_ep_recv_count in shared_expert_recv_count]

            return shared_expert_recv_count_per_card

        def gen_moe_expert_recv_count(
                ep_local_moe_expert_num: torch.Tensor,
                expert_ids: torch.Tensor, x_active_mask: torch.Tensor, available_ranks, table2_elastic, elastic_info
        ):
            x_active_mask_per_card = x_active_mask.chunk(ep_world_size)

            # 将全局的expert_ids切分为ep_world_size份，放入元组expert_ids_segments中。每份为一张卡上的expert_ids
            original_expert_ids_segments = expert_ids.chunk(ep_world_size)
            expert_ids_segments = [
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
                local_moe_expert_recv_count.flatten().cumsum(dim=0, dtype=torch.int32)
                for local_moe_expert_recv_count in moe_expert_recv_count_per_card
            ]

            return moe_expert_recv_count_per_card

        # 拼接共享专家和moe专家收到的token数
        shared_expert_recv_count_per_card = gen_shared_expert_recv_count(x_active_mask_ela_ori, table1_elastic,
                                                                         available_ranks, table2_elastic)
        moe_expert_recv_count_per_card = gen_moe_expert_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask,
                                                                   available_ranks, table2_elastic, elastic_info)
        ep_recv_count = shared_expert_recv_count_per_card + moe_expert_recv_count_per_card

        return ep_recv_count

    if is_zero_compute_expert:
        if len(x_active_mask_list_ela_ori[0].shape) == 1:
            x_active_mask_list_ela_ori = [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in
                                          x_active_mask_list_ela_ori]
    golden_expand_x_list, golden_expand_idx_list, golden_expand_idx_masked_list, golden_tokens_list, \
        golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list, golden_triplets_expand_idx_list, \
        golden_expand_scales_list = [], [], [], [], [], [], [], [], []
    for tp_id in range(tp_world_size):
        ep_local_moe_expert_num, x, expert_ids, expert_scales, scales, x_active_mask, x_active_mask_ela_ori = (
            local_moe_expert_num_list[tp_id],
            x_list[tp_id], expert_ids_list[tp_id], expert_scales_list[tp_id],
            padded_scales_list[tp_id], x_active_mask_list[tp_id], x_active_mask_list_ela_ori[tp_id]
        )

        # region ep域dispatch

        expanded_x, sorted_col_idx, golden_moe_token = moe_dispatch(ep_local_moe_expert_num, x, expert_ids, x_active_mask)

        shared_list, shared_tokens = shared_dispatch(x, x_active_mask_ela_ori, available_ranks, table2_elastic)

        golden_tokens = torch.cat((shared_tokens, golden_moe_token))  # 分到所有卡上的golden值

        moe_x_segments = split_tensor(expanded_x, golden_moe_token.tolist())
        golden_dynamic_scales = []
        if is_quant:
            shared_list, moe_x_segments, golden_dynamic_scales = quant_process(
                ep_local_moe_expert_num,
                shared_list, moe_x_segments, scales,
            )

        if is_noq_cast_hifp8:
            # 场景1：将dispatch路由后的fp16/bf16数据直接cast为hifp8
            from en_dtypes import hifloat8 as hifp8_np_dtype
            for i in range(len(shared_list)):
                if shared_list[i].numel() > 0:
                    np_data = shared_list[i].cpu().float().numpy().astype(hifp8_np_dtype)
                    shared_list[i] = torch.from_numpy(np_data.view(np.uint8).copy())
            for i in range(len(moe_x_segments)):
                if moe_x_segments[i].numel() > 0:
                    np_data = moe_x_segments[i].cpu().float().numpy().astype(hifp8_np_dtype)
                    moe_x_segments[i] = torch.from_numpy(np_data.view(np.uint8).copy())

        if is_noq_forward:
            # 场景2：将per-token scales按照与x相同的dispatch路由进行分发
            x_active_mask_for_scales = x_active_mask
            if is_2d_mask:
                x_active_mask_1d = x_active_mask.any(dim=1)
            else:
                x_active_mask_1d = x_active_mask

            # moe侧scales路由：与moe_dispatch相同的索引
            filtered_scales = scales[x_active_mask_1d]
            dst_row_to_src_row = sorted_col_idx // k
            if is_2d_mask:
                # is_2d_mask时sorted_col_idx的值是全局行号[0, global_bs-1]，
                # 但filtered_scales是compact行（仅real_bs行），直接index_select会越界。
                # 改用全局scales做index_select，再用expanded_mask过滤有效行。
                expanded_mask = torch.index_select(input=x_active_mask.flatten(), dim=0, index=sorted_col_idx)
                expanded_scales = torch.index_select(input=scales, dim=0, index=dst_row_to_src_row)
                expanded_scales = expanded_scales[expanded_mask]
            else:
                expanded_scales = torch.index_select(input=filtered_scales, dim=0, index=dst_row_to_src_row)
            moe_scales_segments = split_tensor(expanded_scales, golden_moe_token.tolist())

            # shared侧scales路由：与shared_dispatch相同的分卡逻辑
            shared_scales_list = [torch.empty(0) for _ in range(shared_expert_rank_num)]
            if shared_expert_rank_num > 0:
                scales_per_card = list(scales.chunk(ep_world_size))
                x_mask_per_card = list(x_active_mask_ela_ori.chunk(ep_world_size))
                if is_2d_mask:
                    real_scales_per_card = [
                        local_scales[local_mask.any(dim=1)]
                        for local_scales, local_mask in zip(scales_per_card, x_mask_per_card)
                    ]
                else:
                    real_scales_per_card = [
                        local_scales[local_mask]
                        for local_scales, local_mask in zip(scales_per_card, x_mask_per_card)
                    ]
                if not is_elastic:
                    for shared_card_id in range(shared_expert_rank_num):
                        card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert
                        send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert)
                        shared_scales_list[shared_card_id] = torch.cat(real_scales_per_card[send_card_ids])
                else:
                    for shared_card_id in range(shared_expert_rank_num_ela):
                        card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert_ela
                        send_card_ids_elastic = list(
                            range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert_ela))
                        send_card_ids = [table2_elastic[i] for i in send_card_ids_elastic]
                        shared_scales_list[shared_card_id] = torch.cat([real_scales_per_card[i] for i in send_card_ids])

            # 组装per-card golden_dynamic_scales
            rank_expert_id_bounds_scales = torch.cat(
                [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]
            ).tolist()
            moe_scales_per_card = [
                torch.cat(moe_scales_segments[rank_expert_id_bounds_scales[mid]: rank_expert_id_bounds_scales[mid + 1]])
                if rank_expert_id_bounds_scales[mid] < rank_expert_id_bounds_scales[mid + 1]
                else torch.empty(0, scales_h, dtype=scales.dtype)
                for mid in range(moe_rank_num)
            ]
            golden_dynamic_scales = shared_scales_list + moe_scales_per_card
        rank_expert_id_bounds = torch.cat(
            [torch.zeros(1, dtype=torch.int32), ep_local_moe_expert_num.cumsum(dim=0)]  # 每个数字, 既是前一张卡的尾，又是后一张卡的头
        ).tolist()
        moe_list = [
            concat_tensors(moe_x_segments[rank_expert_id_bounds[moe_rank_id]: rank_expert_id_bounds[moe_rank_id + 1]])
            for moe_rank_id in range(moe_rank_num)
        ]
        golden_expand_x = shared_list + moe_list

        golden_triplets_expand_idx = gen_triplets_expand_idx(ep_local_moe_expert_num, expert_ids, x_active_mask, x_active_mask_ela_ori, available_ranks, table2_elastic)
        golden_expand_scales = gen_expand_scales(ep_local_moe_expert_num, expert_ids, expert_scales,
                                                 x_active_mask, x_active_mask_ela_ori, available_ranks,
                                                 table2_elastic)
        golden_expand_idx, golden_expand_idx_masked = gen_expand_idx(expert_ids, x_active_mask)
        golden_ep_recv_count = gen_ep_recv_count(ep_local_moe_expert_num, expert_ids, x_active_mask,
                                                 x_active_mask_ela_ori, table1_elastic, available_ranks, table2_elastic,
                                                 elastic_info)

        # endregion

        golden_expand_x_list.append(golden_expand_x)
        golden_expand_idx_list.append(golden_expand_idx)
        golden_expand_idx_masked_list.append(golden_expand_idx_masked)
        golden_tokens_list.append(golden_tokens)
        golden_ep_recv_count_list.append(golden_ep_recv_count)
        golden_dynamic_scales_list.append(golden_dynamic_scales)
        golden_triplets_expand_idx_list.append(golden_triplets_expand_idx)
        golden_expand_scales_list.append(golden_expand_scales)
        middle_idx_list.append(sorted_col_idx)

    return golden_expand_x_list, golden_expand_idx_list, golden_expand_idx_masked_list, \
        golden_tokens_list, golden_ep_recv_count_list, \
        golden_dynamic_scales_list, middle_idx_list, golden_triplets_expand_idx_list, golden_expand_scales_list


def gen_all_gather_golden(
        golden_expand_x_list, golden_tokens_list,
        golden_dynamic_scales_list, golden_ep_recv_count_list,
        tp_world_size
):
    tp_recv_counts = torch.stack([golden_tokens_list[0], golden_tokens_list[1]] * 2, dim=1).view(-1).to(torch.int32)

    golden_tokens = torch.repeat_interleave(golden_tokens_list[0] + golden_tokens_list[1], repeats=2, dim=0)

    gathered_golden_expand_x_list = [[] for _ in range(tp_world_size)]
    expand_x_list = []
    ep_recv_count_list = []
    dynamic_scales_list = []
    for ep_id in range(ep_world_size):
        for tp_id in range(tp_world_size):
            peer_tp_id = (tp_id + 1) % tp_world_size
            gathered_local_expand_x = torch.cat(
                (golden_expand_x_list[tp_id][ep_id], golden_expand_x_list[peer_tp_id][ep_id]))

            gathered_golden_expand_x_list[tp_id].append(gathered_local_expand_x)
            expand_x_list.append(gathered_local_expand_x)
            ep_recv_count_list.append(
                torch.cat((golden_ep_recv_count_list[tp_id][ep_id], golden_ep_recv_count_list[peer_tp_id][ep_id]))
            )
            if is_quant or is_noq_forward:
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
            local_real_token_num = golden_tp_recv_count[2 * ep_id][tp_id]
            peer_real_token_num = golden_tp_recv_count[2 * ep_id][peer_tp_id]
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
        golden_tokens_list ,x_active_mask_list, x_active_mask_list_ela_ori, shared_expert_x_list,
        ori_x_list, expert_ids_list,
        const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
):
    def comm_quant_round_trip(tensor: torch.Tensor):
        """ 将张量x做comm_quant量化，并对量化的结果进行逆运算

            comm_quant_mode=2时，对输入x进行comm_quant操作并返回结果
            comm_quant_mode=0时，直接返回输入x
            comm_quant_mode=3或4时，对输入x进行mxfp8操作并返回
        """

        def comm_quant(tensor: torch.Tensor):
            if combine_mx_quant:
                blocks_np = tensor.view(-1, element_per_block).cpu().float().numpy()
                mx_ele_dtype = 'float8_e5m2' if comm_quant_mode == 3 else 'float8_e4m3fn'
                scale_array, ele_array = mx_quantize(
                    fp_array=blocks_np,
                    mx_ele_dtype=mx_ele_dtype,
                    axis=-1,
                    block_size=element_per_block,
                    round_mode='rint'
                )
                ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
                return numpy_to_torch_tensor(ele_array), scale_array
            else:
                quant_dtype = torch.int8 if comm_quant_mode == 2 else torch.int16
                quant_dtype_max = 127.0 if comm_quant_mode == 2 else 2047.0

                blocks = tensor.view(-1, element_per_block).to(torch.float32)
                blocks_max, _ = blocks.abs().max(dim=-1, keepdim=True)
                blocks_scales = blocks_max / quant_dtype_max
                blocks_scales[blocks_scales == 0.0] = 1.0  # block_scales = 0 的时候改成 1，避免除 0
                blocks_quant = (blocks / blocks_scales).to(torch.float16).round().to(quant_dtype)

                return blocks_quant, blocks_scales.to(combine_input_dtype)

        def comm_dequant(blocks_quant: torch.Tensor, blocks_scales: torch.Tensor):
            if combine_mx_quant:
                mx_dtype = 'float8_e5m2' if comm_quant_mode == 3 else 'float8_e4m3fn'
                tensor_dequant = mx_dequantize(
                    quant_tensor=blocks_quant,
                    scale_tensor=blocks_scales,
                    mx_dtype=mx_dtype,
                    axis=-1
                )
                tensor_dequant = tensor_dequant.view(-1, h + token_pad_length)
            else:
                blocks_quant = blocks_quant.to(torch.float16).to(torch.float32)
                blocks_dequant = (blocks_quant * blocks_scales).to(combine_input_dtype)
                tensor_dequant = blocks_dequant.view(-1, h + token_pad_length)

            return tensor_dequant

        if comm_quant_mode == 0:
            return tensor

        block_size = 32  # 核函数侧，一个 block 有 32 字节
        computation_dtype = torch.float32

        computation_dtype_size = torch.finfo(computation_dtype).bits // 8  # 用于量化计算的数据类型的字节数
        element_per_block = 32 if combine_mx_quant else (block_size // computation_dtype_size)  # 8

        token_pad_length = (element_per_block - (tensor.size(1) % element_per_block)) % element_per_block
        padded_tensor = torch.cat([tensor, tensor.new_zeros(tensor.size(0), token_pad_length)], dim=1)
        padded_tensor_after_round_trip = comm_dequant(*comm_quant(padded_tensor))

        return padded_tensor_after_round_trip[:, :tensor.size(1)]

    def combine_x_for_shared(
            middle_shared_x_per_card: List[torch.Tensor],
            x_active_mask: torch.Tensor, shared_scales_list=[1 for _ in range(shared_expert_num)]
    ):
        returned_real_x = [[torch.empty(0) for _ in range(ep_world_size)] for _ in range(shared_expert_num)]
        if is_2d_mask:
            real_token_num_list = x_active_mask.view(ep_world_size, bs, -1).any(dim=-1).sum(dim=1).tolist()
        else:
            real_token_num_list = x_active_mask.view(-1, bs).sum(dim=1).tolist()

        for shared_card_id in range(shared_expert_rank_num_ela):
            shared_expert_index = shared_card_id // rank_num_per_shared_expert_ela

            card_id_in_shared_group = shared_card_id % rank_num_per_shared_expert_ela
            if not is_elastic:
                send_card_ids = slice(card_id_in_shared_group, ep_world_size, rank_num_per_shared_expert_ela)

                local_middle_shared_x = middle_shared_x_per_card[shared_card_id]
                token_num_from_origin_card = real_token_num_list[send_card_ids]

                returned_real_x[shared_expert_index][send_card_ids] = split_tensor(local_middle_shared_x,
                                                                                   token_num_from_origin_card)
            else:
                send_card_ids_elastic = list(
                    range(card_id_in_shared_group, len(available_ranks), rank_num_per_shared_expert_ela))
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

        # is_noq_special时middle_moe_x和ori_x是特殊编码（fp4 packed / hifp8 uint8），需先反量化为float
        if is_noq_special:
            middle_moe_x = convert_expand_x_dtype(middle_moe_x)
            ori_x = convert_expand_x_dtype(ori_x)

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
                    scaled_x[token_id_global] += x_k_after_round_trip[expand_indx] * expert_scales[token_id_global][
                        top_id]
                    expand_indx += 1

                elif expert_ids[token_id_global][top_id] < moe_expert_num + zero_expert_num:
                    continue
                elif expert_ids[token_id_global][top_id] < moe_expert_num + zero_expert_num + copy_expert_num:
                    scaled_x[token_id_global] += ori_x_32[token_id_global] * expert_scales[token_id_global][top_id]
                else:
                    alpha_1 = const_expert_alpha_1_32[expert_ids[token_id_global][top_id].item() \
                                                                    - moe_expert_num - zero_expert_num - copy_expert_num]
                    alpha_2 = const_expert_alpha_2_32[expert_ids[token_id_global][top_id].item() \
                                                                    - moe_expert_num - zero_expert_num - copy_expert_num]
                    alpha_1 = alpha_1 * ori_x_32[token_id_global]
                    alpha_2 = alpha_2 * ori_x_32[token_id_global]

                    alpha1_sum = binary_tree_sum(alpha_1)
                    alpha2_sum = binary_tree_sum(alpha_2)

                    alpha = torch.tensor([alpha1_sum, alpha2_sum])

                    max = torch.max(alpha)
                    alpha_sum = torch.exp((alpha[0] - max).double()).float() + torch.exp((alpha[1] - max).double()).float()
                    alpha1 = torch.exp((alpha[0] - max).double()).float() / alpha_sum
                    alpha2 = torch.exp((alpha[1] - max).double()).float() / alpha_sum

                    v = const_expert_v_32[expert_ids[token_id_global][top_id].item() - moe_expert_num - zero_expert_num - copy_expert_num]
                    scaled_x[token_id_global] += (alpha1 * ori_x_32[token_id_global] + alpha2 * v) * expert_scales[token_id_global][top_id]

        scaled_x = torch.stack(scaled_x, dim=0)

        return scaled_x[x_active_mask] if x_active_mask.dim() == 1 else scaled_x

    if is_zero_compute_expert:
        if len(x_active_mask_list_ela_ori[0].shape) == 1:
            x_active_mask_list_ela_ori = [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in
                                          x_active_mask_list_ela_ori]
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
                    shared_expert_x_per_card_vb.append(
                        F.pad(shared_expert_x_per_card[i], (0, 0, 0, bs - shared_expert_x_per_card[i].size(0))))
            if is_elastic and not is_2d_mask:
                shared_expert_x_per_card = [shared_expert_x_per[mask_shared_x[:len(shared_expert_x_per)]] for
                                            shared_expert_x_per, mask_shared_x in zip(shared_expert_x_per_card,
                                                                                      chunk_tensor(
                                                                                          x_active_mask_ela_ori,
                                                                                          ep_world_size))]
            golden_x_ep += torch.cat(
                shared_expert_x_per_card if not shared_expert_x_per_card_vb else shared_expert_x_per_card_vb).to(
                torch.float32)

        if is_noq_special:
            golden_x_ep = golden_x_ep.to(combine_input_dtype)
        else:
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

def compare_output(compared_output: ComparedOutput, golden_per_card, npu_per_card, compare_dtype, compare_per_rank,
                   available_ranks, skip_rank=None):
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
        elif dtype in ("float4_e2m1", "float4_e1m2"):
            # fp4 packed as uint8，直接按字节比较
            return tensor.view(torch.uint8).numpy()
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
        if dtype in ("int8", "float32", "float8_e5m2", "float8_e4m3fn", "float8_e8m0"):
            max_diff_hd = 1.0
        res_output = toolss.data_compare_np(np_output_npu, np_output_golden, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)
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
        for i in (
        range(len(available_ranks)) if (is_elastic and server_num == 1) else range(server_index * rank_per_dev,
                                                                                   (server_index + 1) * rank_per_dev)):
            print(f"##################Compare {mark_str} in rank", i, "epId", i // tp_world_size, "tpId", i % tp_world_size, "##################", flush=True)
            local_rank_id = i - server_index * rank_per_dev
            print(f'local_rank_id={local_rank_id}', flush=True)
            if skip_rank is not None and skip_rank(i, local_rank_id):
                print(f"Skip {mark_str} in rank {i}", flush=True)
                continue

            res = compare(
                golden_per_card[local_rank_id],
                npu_per_card[local_rank_id],
                compare_dtype,
                f"{compared_output.value}_{i}"
            )
            result[f"{compared_output.value}_{i}"] = res

    return result


# endregion

CYCLES_PER_US = globals().get("profile_cycles_per_us", 1000.0)
PROFILER_MAX_CORE_NUM = 72
PROFILER_MAX_EVENT_NUM = 96
PROFILER_RECORD_SIZE = 16
PROFILER_HEADER_SIZE = 64
PROFILER_CORE_COUNT_ALIGN = 64
PROFILER_DATA_OFFSET = PROFILER_HEADER_SIZE + PROFILER_MAX_CORE_NUM * PROFILER_CORE_COUNT_ALIGN
PROFILER_BUFFER_SIZE = PROFILER_DATA_OFFSET + (
    PROFILER_MAX_CORE_NUM * PROFILER_MAX_EVENT_NUM * PROFILER_RECORD_SIZE
)

MAGIC = 0x50524F46
HEADER_SIZE = 64
EVENT_NAMES = {
    0x0001: "KERNEL",
    0x0010: "SEND_TOKENS",
    0x0011: "SEND_TOKENS_END",
    0x0020: "SEND_COUNTS",
    0x0021: "SEND_COUNTS_END",
    0x0030: "WAIT_RECV_COUNTS",
    0x0031: "WAIT_RECV_COUNTS_END",
    0x0040: "LOCAL_WINDOW_COPY",
    0x0041: "LOCAL_WINDOW_COPY_END",
    0x0050: "WAIT_CUMSUM",
    0x0051: "WAIT_CUMSUM_END",
    0x0060: "FORMAT_OUTPUT",
    0x0061: "FORMAT_OUTPUT_END",
    0x0070: "PREPARE_TOKEN_INPUT",
    0x0071: "PREPARE_TOKEN_INPUT_END",
    0x0080: "TRANSFER_TOKENS",
    0x0081: "TRANSFER_TOKENS_END",
    0x0090: "PREPARE_COUNT_INPUT",
    0x0091: "PREPARE_COUNT_INPUT_END",
    0x00A0: "CLEAR_RECV_STATUS",
    0x00A1: "CLEAR_RECV_STATUS_END",
    0x00B0: "GATHER_RECV_COUNTS",
    0x00B1: "GATHER_RECV_COUNTS_END",
    0x00C0: "GET_CUMSUM",
    0x00C1: "GET_CUMSUM_END",
    0x00D0: "BUILD_RECV_CUMSUM",
    0x00D1: "BUILD_RECV_CUMSUM_END",
    0x00E0: "PUBLISH_RECV_COUNTS",
    0x00E1: "PUBLISH_RECV_COUNTS_END",
    0x0100: "RESET_CUMSUM_FLAG",
    0x0101: "RESET_CUMSUM_FLAG_END",
    0x0110: "BUILD_EXPERT_TASKS",
    0x0111: "BUILD_EXPERT_TASKS_END",
    0x0120: "CLEAR_LOCAL_FLAGS",
    0x0121: "CLEAR_LOCAL_FLAGS_END",
    0x0130: "DRAIN_PIPELINES",
    0x0131: "DRAIN_PIPELINES_END",
    0x0140: "NO_ACTIVE_TOKENS",
    0x0141: "NO_ACTIVE_TOKENS_END",
    0x0150: "NO_ASSIGNED_STATUS",
    0x0151: "NO_ASSIGNED_STATUS_END",
    0x0160: "NO_VALID_EXPERT",
    0x0161: "NO_VALID_EXPERT_END",
    0x0170: "SEND_PROFILE_MODE",
    0x0171: "SEND_TASK_COUNT",
    0x0172: "SEND_POSITION_CYCLES",
    0x0173: "SEND_INPUT_CYCLES",
    0x0174: "SEND_QUANT_CYCLES",
    0x0175: "SEND_BUFFER_WAIT_CYCLES",
    0x0180: "RECV_ROUND_BEGIN",
    0x0181: "RECV_ROUND_END",
    0x0190: "SEND_TOKEN",
    0x0191: "SEND_TOKEN_END",
    0x01A0: "RECV_SLOT_ARRIVE",
    0x00FF: "KERNEL_END",
    0x0200: "CAL_CUMSUM",
    0x0201: "CAL_CUMSUM_END",
    0x0210: "INIT_COUNT_STATUS",
    0x0211: "INIT_COUNT_STATUS_END",
    0x0220: "COMPUTE_EXPERT_COUNTS",
    0x0221: "COMPUTE_EXPERT_COUNTS_END",
    0x0230: "TRANSFER_COUNTS",
    0x0231: "TRANSFER_COUNTS_END",
    0x0240: "POLL_RECV_COUNTS",
    0x0241: "POLL_RECV_COUNTS_END",
    0x0250: "REDUCE_LOCAL_COUNTS",
    0x0251: "REDUCE_LOCAL_COUNTS_END",
    0x0260: "PUBLISH_LOCAL_SUM",
    0x0261: "PUBLISH_LOCAL_SUM_END",
    0x0270: "POLL_CORE_SUMS",
    0x0271: "POLL_CORE_SUMS_END",
    0x0280: "REDUCE_CORE_PREFIX",
    0x0281: "REDUCE_CORE_PREFIX_END",
    0x0290: "CLEAR_CORE_SUMS",
    0x0291: "CLEAR_CORE_SUMS_END",
    0x02A0: "WRITE_COUNTS_OUTPUT",
    0x02A1: "WRITE_COUNTS_OUTPUT_END",
    0x02B0: "BROADCAST_PREFIX",
    0x02B1: "BROADCAST_PREFIX_END",
    0x02C0: "PUBLISH_PREFIX_FLAGS",
    0x02C1: "PUBLISH_PREFIX_FLAGS_END",
    0x02D0: "READ_PREFIX",
    0x02D1: "READ_PREFIX_END",
    0x02E0: "DRAIN_SEND_BUFFERS",
    0x02E1: "DRAIN_SEND_BUFFERS_END",
    0x02F0: "INIT_COUNT_BUFFERS",
    0x02F1: "INIT_COUNT_BUFFERS_END",
    0x0300: "SEND_METADATA_CYCLES",
    0x0301: "SEND_PACK_CYCLES",
    0x0302: "SEND_MTE3_ISSUE_CYCLES",
    0x0303: "COUNT_COMPUTE_CYCLES",
    0x0304: "COUNT_ISSUE_CYCLES",
    0x0305: "SEND_INPUT_COUNT",
    0x0306: "SEND_QUANT_COUNT",
    0xFFFF: "OVERFLOW",
}
PAIR_MAP = {
    0x0200: 0x0201,
    0x0210: 0x0211,
    0x0220: 0x0221,
    0x0230: 0x0231,
    0x0240: 0x0241,
    0x0250: 0x0251,
    0x0260: 0x0261,
    0x0270: 0x0271,
    0x0280: 0x0281,
    0x0290: 0x0291,
    0x02A0: 0x02A1,
    0x02B0: 0x02B1,
    0x02C0: 0x02C1,
    0x02D0: 0x02D1,
    0x02E0: 0x02E1,
    0x02F0: 0x02F1,

    0x0001: 0x00FF,
    0x0010: 0x0011,
    0x0020: 0x0021,
    0x0030: 0x0031,
    0x0040: 0x0041,
    0x0050: 0x0051,
    0x0060: 0x0061,
    0x0070: 0x0071,
    0x0080: 0x0081,
    0x0090: 0x0091,
    0x00A0: 0x00A1,
    0x00B0: 0x00B1,
    0x00C0: 0x00C1,
    0x00D0: 0x00D1,
    0x00E0: 0x00E1,
    0x0100: 0x0101,
    0x0110: 0x0111,
    0x0120: 0x0121,
    0x0130: 0x0131,
    0x0140: 0x0141,
    0x0150: 0x0151,
    0x0160: 0x0161,
    0x0180: 0x0181,
    0x0190: 0x0191,
}
COLORS = {
    "CAL_CUMSUM": "#6a1b9a",
    "INIT_COUNT_STATUS": "#8e24aa",
    "COMPUTE_EXPERT_COUNTS": "#ab47bc",
    "TRANSFER_COUNTS": "#ce93d8",
    "POLL_RECV_COUNTS": "#ef6c00",
    "REDUCE_LOCAL_COUNTS": "#f9a825",
    "PUBLISH_LOCAL_SUM": "#fb8c00",
    "POLL_CORE_SUMS": "#d84315",
    "REDUCE_CORE_PREFIX": "#3949ab",
    "CLEAR_CORE_SUMS": "#5e35b1",
    "WRITE_COUNTS_OUTPUT": "#1e88e5",
    "BROADCAST_PREFIX": "#1565c0",
    "PUBLISH_PREFIX_FLAGS": "#0d47a1",
    "READ_PREFIX": "#00897b",
    "DRAIN_SEND_BUFFERS": "#455a64",
    "INIT_COUNT_BUFFERS": "#78909c",

    "KERNEL": "#455a64",
    "SEND_TOKENS": "#1976d2",
    "SEND_COUNTS": "#7b1fa2",
    "WAIT_RECV_COUNTS": "#ef6c00",
    "LOCAL_WINDOW_COPY": "#388e3c",
    "WAIT_CUMSUM": "#d32f2f",
    "FORMAT_OUTPUT": "#00838f",
    "PREPARE_TOKEN_INPUT": "#64b5f6",
    "TRANSFER_TOKENS": "#0d47a1",
    "PREPARE_COUNT_INPUT": "#ba68c8",
    "CLEAR_RECV_STATUS": "#ffb300",
    "GATHER_RECV_COUNTS": "#e64a19",
    "GET_CUMSUM": "#5e35b1",
    "BUILD_RECV_CUMSUM": "#3949ab",
    "PUBLISH_RECV_COUNTS": "#283593",
    "RESET_CUMSUM_FLAG": "#ad1457",
    "BUILD_EXPERT_TASKS": "#7cb342",
    "CLEAR_LOCAL_FLAGS": "#00695c",
    "DRAIN_PIPELINES": "#78909c",
    "NO_ACTIVE_TOKENS": "#9e9e9e",
    "NO_ASSIGNED_STATUS": "#757575",
    "NO_VALID_EXPERT": "#424242",
    "RECV_ROUND_BEGIN": "#c62828",
    "SEND_TOKEN": "#1565c0",
}
STAGE_DESCRIPTIONS = {
    "CAL_CUMSUM": 'CalCumSum: complete count path before Process drain',
    "INIT_COUNT_STATUS": 'Initialize the existing full expert status UB',
    "COMPUTE_EXPERT_COUNTS": 'CalExpertCnt loop (elastic path includes interleaved sends; see COUNT_COMPUTE_CYCLES)',
    "TRANSFER_COUNTS": 'Issue expert counts to remote status windows, including existing waits',
    "POLL_RECV_COUNTS": 'Poll remote count-ready flags only; end payload is poll count',
    "REDUCE_LOCAL_COUNTS": 'Gather and reduce this core received counts',
    "PUBLISH_LOCAL_SUM": 'Replicate and publish local sum to the C-by-C matrix, including existing drain',
    "POLL_CORE_SUMS": 'Poll all cumsum cores local sums; end payload is poll count',
    "REDUCE_CORE_PREFIX": 'Reduce sums preceding this core',
    "CLEAR_CORE_SUMS": 'Issue clears for this consumer column of the local-sum matrix',
    "WRITE_COUNTS_OUTPUT": 'Issue public sendCounts output',
    "BROADCAST_PREFIX": 'Issue A copies of this core prefix fragment to local workspace',
    "PUBLISH_PREFIX_FLAGS": 'Prepare/publish prefix completion flags, including existing MTE3 barrier',
    "READ_PREFIX": 'Read this core local workspace prefix copy, including existing PIPE_ALL',
    "DRAIN_SEND_BUFFERS": 'Existing final waits for all send buffers',
    "INIT_COUNT_BUFFERS": 'Allocate and initialize cumsum input buffers',

    "KERNEL": "Process total",
    "SEND_TOKENS": "AllToAllDispatch: send token data",
    "SEND_COUNTS": "CalAndSendCnt*: calculate and send token counts",
    "WAIT_RECV_COUNTS": "WaitDispatch: wait, clear status and sum received counts",
    "LOCAL_WINDOW_COPY": "LocalWindowCopy: compact local expert output",
    "WAIT_CUMSUM": "WaitCumSumFlag: wait for count prefix sum",
    "FORMAT_OUTPUT": "WaitAndFormatOutput: active cores write expert output",
    "PREPARE_TOKEN_INPUT": "ExpIdsCopyAndMaskCal: prepare token dispatch input",
    "TRANSFER_TOKENS": "AllToAllDispatch: copy token data to remote windows",
    "PREPARE_COUNT_INPUT": "ExpIdsCopyAndMaskCal: prepare count calculation input",
    "CLEAR_RECV_STATUS": "WaitDispatchClearStatus: clear received status flags",
    "GATHER_RECV_COUNTS": "GatherSumRecvCnt: sum per-core received counts",
    "GET_CUMSUM": "GetCumSum: calculate prefix across count cores",
    "BUILD_RECV_CUMSUM": "CalRecvAndSetFlag: build local received-count prefix sum",
    "PUBLISH_RECV_COUNTS": "CalRecvAndSetFlag: publish counts and completion flags",
    "RESET_CUMSUM_FLAG": "WaitCumSumFlag: record progress and clear sync flags",
    "BUILD_EXPERT_TASKS": "SetValidExpertInfo: select experts with received tokens",
    "CLEAR_LOCAL_FLAGS": "ClearLocalWindowDataFlags: clear consumed data flags",
    "DRAIN_PIPELINES": "PipeBarrier: wait for previous pipeline operations",
    "NO_ACTIVE_TOKENS": "AllToAllDispatch: this core has no active token to send",
    "NO_ASSIGNED_STATUS": "LocalWindowCopy: this core has no assigned status",
    "NO_VALID_EXPERT": "LocalWindowCopy: assigned experts received no token",
    "RECV_ROUND_BEGIN": "LocalWindowCopy: activate one token receive context",
    "SEND_TOKEN": "AllToAllDispatch: one token copy to the remote window",
}
TIMELINE_STAGES = (
    "CAL_CUMSUM",
    "INIT_COUNT_STATUS",
    "COMPUTE_EXPERT_COUNTS",
    "TRANSFER_COUNTS",
    "POLL_RECV_COUNTS",
    "REDUCE_LOCAL_COUNTS",
    "PUBLISH_LOCAL_SUM",
    "POLL_CORE_SUMS",
    "REDUCE_CORE_PREFIX",
    "CLEAR_CORE_SUMS",
    "WRITE_COUNTS_OUTPUT",
    "BROADCAST_PREFIX",
    "PUBLISH_PREFIX_FLAGS",
    "READ_PREFIX",
    "DRAIN_SEND_BUFFERS",
    "INIT_COUNT_BUFFERS",

    "SEND_TOKENS",
    "PREPARE_TOKEN_INPUT",
    "TRANSFER_TOKENS",
    "SEND_COUNTS",
    "PREPARE_COUNT_INPUT",
    "WAIT_RECV_COUNTS",
    "CLEAR_RECV_STATUS",
    "GATHER_RECV_COUNTS",
    "GET_CUMSUM",
    "BUILD_RECV_CUMSUM",
    "PUBLISH_RECV_COUNTS",
    "DRAIN_PIPELINES",
    "WAIT_CUMSUM",
    "RESET_CUMSUM_FLAG",
    "LOCAL_WINDOW_COPY",
    "BUILD_EXPERT_TASKS",
    "FORMAT_OUTPUT",
    "CLEAR_LOCAL_FLAGS",
    "NO_ACTIVE_TOKENS",
    "NO_ASSIGNED_STATUS",
    "NO_VALID_EXPERT",
    "RECV_ROUND_BEGIN",
    "SEND_TOKEN",
)
SEND_METRIC_FIELDS = {
    0x0170: "mode",
    0x0171: "tasks",
    0x0172: "position_cycles",
    0x0173: "input_cycles",
    0x0174: "quant_cycles",
    0x0175: "buffer_wait_cycles",
}
SEND_SUMMARY_FIELDS = (
    "mode",
    "rank",
    "core_id",
    "tasks",
    "transfer_cycles",
    "position_cycles",
    "input_cycles",
    "quant_cycles",
    "buffer_wait_cycles",
)
RECV_ROUND_BEGIN_EVENT = 0x0180
RECV_SLOT_ARRIVE_EVENT = 0x01A0
RECV_ROUND_FIELDS = (
    "rank",
    "core_id",
    "ready_order",
    "round_id",
    "token_id",
    "start_cycle",
    "start_us",
    "gap_cycles",
    "gap_us",
    "format_version",
)
RECV_TOKEN_FIELDS = (
    "rank",
    "core_id",
    "token_local_idx",
    "token_id",
    "start_cycle",
    "end_cycle",
    "duration_cycles",
    "start_us",
    "end_us",
    "duration_us",
    "format_version",
)
SEND_TOKEN_FIELDS = (
    "rank",
    "core_id",
    "send_order",
    "to_rank_id",
    "token_id",
    "topk_id",
    "tk_index",
    "start_cycle",
    "end_cycle",
    "duration_cycles",
    "start_us",
    "end_us",
    "duration_us",
    "format_version",
)
RECV_SLOT_ARRIVE_FIELDS = (
    "rank",
    "rank_arrival_order",
    "core_id",
    "core_arrival_order",
    "token_id",
    "slot_begin",
    "slot_end",
    "slot_count",
    "observed_cycle",
    "observed_us",
    "core_gap_cycles",
    "core_gap_us",
    "format_version",
)


def rank_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"rank(\d+)", path.name)
    return int(match.group(1)) if match else fallback


def parse_binary(path: Path, rank: int) -> list[dict[str, int | str]]:
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"{path}: file is shorter than the 64-byte header")
    header = struct.unpack_from("<8I", data, 0)
    magic, version_layout, core_num, record_size, capacity, count_stride, data_offset, _ = header
    if magic != MAGIC:
        raise ValueError(f"{path}: invalid magic 0x{magic:08X}")
    if not 1 <= core_num <= PROFILER_MAX_CORE_NUM or not 1 <= capacity <= 4096:
        raise ValueError(f"{path}: invalid core count/capacity {core_num}/{capacity}")
    if count_stride < 4 or data_offset < HEADER_SIZE + core_num * count_stride:
        raise ValueError(f"{path}: overlapping header/count/record regions")
    if (version_layout & 0xFFFF) not in (2, 3, 4) or (version_layout >> 16) != 1:
        raise ValueError(f"{path}: unsupported profiler version/layout 0x{version_layout:08X}")
    if record_size != 16:
        raise ValueError(f"{path}: unsupported record size {record_size}")
    expected_size = data_offset + core_num * capacity * record_size
    if len(data) < expected_size:
        raise ValueError(f"{path}: got {len(data)} bytes, expected at least {expected_size}")

    rows: list[dict[str, int | str]] = []
    for core_id in range(core_num):
        count_offset = HEADER_SIZE + core_id * count_stride
        count = struct.unpack_from("<I", data, count_offset)[0]
        if count == 0:
            raise ValueError(f"{path}: core {core_id} has no records; dump is incomplete")
        if count > capacity:
            raise ValueError(f"{path}: core {core_id} count {count} exceeds capacity {capacity}")
        core_offset = data_offset + core_id * capacity * record_size
        for index in range(count):
            event_id, payload, cycle = struct.unpack_from("<IIQ", data, core_offset + index * record_size)
            if event_id == 0xFFFF:
                raise ValueError(f"{path}: core {core_id} profiler overflow; increase capacity and recollect")
            rows.append({
                "rank": rank,
                "core_id": core_id,
                "index": index,
                "event_id": event_id,
                "event_name": EVENT_NAMES.get(event_id, f"EVT_0x{event_id:04X}"),
                "payload": payload,
                "cycle": cycle,
                "format_version": version_layout & 0xFFFF,
                "layout": version_layout >> 16,
            })
    return rows


def pair_events(rows: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    by_lane: dict[tuple[int, int], list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        by_lane[(int(row["rank"]), int(row["core_id"]))].append(row)

    pairs: list[dict[str, int | str]] = []
    end_to_begin = {end: begin for begin, end in PAIR_MAP.items()}
    for (rank, core_id), lane_rows in sorted(by_lane.items()):
        pending: dict[int, deque[dict[str, int | str]]] = defaultdict(deque)
        for row in sorted(lane_rows, key=lambda item: int(item["index"])):
            event_id = int(row["event_id"])
            if event_id in PAIR_MAP:
                pending[event_id].append(row)
            elif event_id in end_to_begin:
                begin_id = end_to_begin[event_id]
                if not pending[begin_id]:
                    raise ValueError(f"rank {rank} core {core_id}: unmatched end event {event_id:#x}")
                begin = pending[begin_id].popleft()
                begin_cycle = int(begin["cycle"])
                end_cycle = int(row["cycle"])
                if end_cycle < begin_cycle:
                    raise ValueError(f"rank {rank} core {core_id}: reversed cycle interval")
                pairs.append({
                    "rank": rank,
                    "core_id": core_id,
                    "stage": EVENT_NAMES[begin_id],
                    "begin_cycle": begin_cycle,
                    "end_cycle": end_cycle,
                    "duration_cycles": end_cycle - begin_cycle,
                    "payload": int(begin["payload"]),
                    "begin_payload": int(begin["payload"]),
                    "end_payload": int(row["payload"]),
                    "format_version": int(begin["format_version"]),
                })
        if any(pending.values()):
            raise ValueError(f"rank {rank} core {core_id}: unclosed profiler intervals")
    return pairs


def build_core_summary(pairs: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    by_lane: dict[tuple[int, int], dict[str, dict[str, int | str]]] = defaultdict(dict)
    for pair in pairs:
        lane = (int(pair["rank"]), int(pair["core_id"]))
        by_lane[lane][str(pair["stage"])] = pair

    def field(stages: dict[str, dict[str, int | str]], stage: str, name: str) -> int | str:
        return stages[stage][name] if stage in stages else ""

    summaries: list[dict[str, int | str]] = []
    for (rank, core_id), stages in sorted(by_lane.items()):
        format_version = int(field(stages, "KERNEL", "format_version") or 0)
        has_detailed_payload = format_version >= 2
        kernel_payload = int(field(stages, "KERNEL", "begin_payload") or 0)
        send_token_cores = kernel_payload >> 16
        format_end = field(stages, "FORMAT_OUTPUT", "end_cycle")
        no_valid_end = field(stages, "NO_VALID_EXPERT", "end_cycle")
        no_status_end = field(stages, "NO_ASSIGNED_STATUS", "end_cycle")
        local_end = field(stages, "LOCAL_WINDOW_COPY", "end_cycle")
        local_work_end = format_end or no_valid_end or no_status_end
        local_tail_cycles: int | str = ""
        if local_end != "" and local_work_end != "":
            local_tail_cycles = int(local_end) - int(local_work_end)
        summaries.append({
            "rank": rank,
            "core_id": core_id,
            "format_version": format_version,
            "role": "send_token" if core_id < send_token_cores else "cumsum",
            "kernel_cycles": field(stages, "KERNEL", "duration_cycles"),
            "kernel_end_cycle": field(stages, "KERNEL", "end_cycle"),
            "wait_recv_cycles": field(stages, "WAIT_RECV_COUNTS", "duration_cycles"),
            "wait_recv_polls": field(stages, "POLL_RECV_COUNTS" if format_version >= 4 else "WAIT_RECV_COUNTS", "end_payload") if has_detailed_payload else "",
            "get_cumsum_cycles": field(stages, "GET_CUMSUM", "duration_cycles"),
            "get_cumsum_polls": field(stages, "POLL_CORE_SUMS" if format_version >= 4 else "GET_CUMSUM", "end_payload") if has_detailed_payload else "",
            "publish_end_cycle": field(stages, "PUBLISH_RECV_COUNTS", "end_cycle"),
            "drain_end_cycle": field(stages, "DRAIN_PIPELINES", "end_cycle"),
            "wait_cumsum_cycles": field(stages, "WAIT_CUMSUM", "duration_cycles"),
            "wait_cumsum_polls": field(stages, "WAIT_CUMSUM", "end_payload") if has_detailed_payload else "",
            "wait_cumsum_end_cycle": field(stages, "WAIT_CUMSUM", "end_cycle"),
            "assigned_status": field(stages, "BUILD_EXPERT_TASKS", "begin_payload"),
            "valid_status": field(stages, "BUILD_EXPERT_TASKS", "end_payload"),
            "assigned_tokens": field(stages, "FORMAT_OUTPUT", "end_payload") if has_detailed_payload and format_version < 4 else "",
            "copy_batches": field(stages, "CLEAR_LOCAL_FLAGS", "end_payload") if has_detailed_payload else "",
            "format_cycles": field(stages, "FORMAT_OUTPUT", "duration_cycles"),
            "format_end_cycle": format_end,
            "local_window_cycles": field(stages, "LOCAL_WINDOW_COPY", "duration_cycles"),
            "local_window_end_cycle": local_end,
            "local_tail_cycles": local_tail_cycles,
        })
    return summaries


def build_send_summary(rows: list[dict[str, int | str]],
                       pairs: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    lane_metrics: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        event_id = int(row["event_id"])
        if event_id not in SEND_METRIC_FIELDS:
            continue
        lane = (int(row["rank"]), int(row["core_id"]))
        lane_metrics[lane][SEND_METRIC_FIELDS[event_id]] = int(row["payload"])

    transfer_pairs = [pair for pair in pairs if pair["stage"] == "TRANSFER_TOKENS"]
    transfer_pairs = [pair for pair in transfer_pairs
                      if (int(pair["rank"]), int(pair["core_id"])) in lane_metrics]
    if not transfer_pairs:
        return []

    slowest = max(transfer_pairs, key=lambda pair: int(pair["duration_cycles"]))
    rank = int(slowest["rank"])
    core_id = int(slowest["core_id"])
    metrics = lane_metrics[(rank, core_id)]
    mode_names = {
        0: "normal",
        1: "bypass_quant",
        2: "pergroup_load_only",
    }
    mode = mode_names.get(metrics.get("mode", 0), f"unknown_{metrics.get('mode', 0)}")
    return [{
        "mode": mode,
        "rank": rank,
        "core_id": core_id,
        "tasks": metrics.get("tasks", ""),
        "transfer_cycles": int(slowest["duration_cycles"]),
        "position_cycles": metrics.get("position_cycles", ""),
        "input_cycles": metrics.get("input_cycles", ""),
        "quant_cycles": metrics.get("quant_cycles", ""),
        "buffer_wait_cycles": metrics.get("buffer_wait_cycles", ""),
    }]


def build_recv_rounds(rows: list[dict[str, int | str]]) -> list[dict[str, int | str | float]]:
    lane_rows: dict[tuple[int, int], list[dict[str, int | str]]] = defaultdict(list)
    wait_begin_cycles: dict[tuple[int, int], int] = {}
    for row in rows:
        lane = (int(row["rank"]), int(row["core_id"]))
        event_id = int(row["event_id"])
        if event_id == RECV_ROUND_BEGIN_EVENT:
            lane_rows[lane].append(row)
        elif event_id == 0x0030:
            wait_begin_cycles[lane] = int(row["cycle"])

    recv_rounds: list[dict[str, int | str | float]] = []
    for (rank, core_id), rows_per_lane in sorted(lane_rows.items()):
        previous_cycle = wait_begin_cycles.get((rank, core_id), int(rows_per_lane[0]["cycle"]))
        for ready_order, row in enumerate(sorted(rows_per_lane, key=lambda item: int(item["index"]))):
            payload = int(row["payload"])
            start_cycle = int(row["cycle"])
            gap_cycles = start_cycle - previous_cycle
            recv_rounds.append({
                "rank": rank,
                "core_id": core_id,
                "ready_order": ready_order,
                "round_id": payload >> 16,
                "token_id": payload & 0xFFFF,
                "start_cycle": start_cycle,
                "start_us": start_cycle / CYCLES_PER_US,
                "gap_cycles": gap_cycles,
                "gap_us": gap_cycles / CYCLES_PER_US,
                "format_version": int(row["format_version"]),
            })
            previous_cycle = start_cycle
    return recv_rounds


def build_recv_token_rounds(pairs: list[dict[str, int | str]]) -> list[dict[str, int | str | float]]:
    recv_token_rounds = []
    for pair in pairs:
        if str(pair["stage"]) != "RECV_ROUND_BEGIN":
            continue
        start_cycle = int(pair["begin_cycle"])
        end_cycle = int(pair["end_cycle"])
        duration_cycles = int(pair["duration_cycles"])
        payload = int(pair["begin_payload"])
        recv_token_rounds.append({
            "rank": int(pair["rank"]),
            "core_id": int(pair["core_id"]),
            "token_local_idx": payload >> 16,
            "token_id": payload & 0xFFFF,
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "duration_cycles": duration_cycles,
            "start_us": start_cycle / CYCLES_PER_US,
            "end_us": end_cycle / CYCLES_PER_US,
            "duration_us": duration_cycles / CYCLES_PER_US,
            "format_version": int(pair["format_version"]),
        })
    return recv_token_rounds


def build_recv_slot_arrivals(rows: list[dict[str, int | str]]) -> list[dict[str, int | str | float]]:
    arrival_rows = sorted(
        (row for row in rows if int(row["event_id"]) == RECV_SLOT_ARRIVE_EVENT),
        key=lambda row: (int(row["rank"]), int(row["cycle"]), int(row["core_id"]), int(row["index"])),
    )
    rank_arrival_order: dict[int, int] = defaultdict(int)
    core_arrival_order: dict[tuple[int, int], int] = defaultdict(int)
    previous_core_cycle: dict[tuple[int, int], int] = {}
    arrivals = []
    for row in arrival_rows:
        rank = int(row["rank"])
        core_id = int(row["core_id"])
        lane = (rank, core_id)
        payload = int(row["payload"])
        token_id = payload >> 16
        slot_begin = (payload >> 8) & 0xFF
        slot_count = payload & 0xFF
        observed_cycle = int(row["cycle"])
        previous_cycle = previous_core_cycle.get(lane)
        core_gap_cycles: int | str = "" if previous_cycle is None else observed_cycle - previous_cycle
        arrivals.append({
            "rank": rank,
            "rank_arrival_order": rank_arrival_order[rank],
            "core_id": core_id,
            "core_arrival_order": core_arrival_order[lane],
            "token_id": token_id,
            "slot_begin": slot_begin,
            "slot_end": slot_begin + slot_count - 1,
            "slot_count": slot_count,
            "observed_cycle": observed_cycle,
            "observed_us": observed_cycle / CYCLES_PER_US,
            "core_gap_cycles": core_gap_cycles,
            "core_gap_us": "" if core_gap_cycles == "" else int(core_gap_cycles) / CYCLES_PER_US,
            "format_version": int(row["format_version"]),
        })
        rank_arrival_order[rank] += 1
        core_arrival_order[lane] += 1
        previous_core_cycle[lane] = observed_cycle
    return arrivals


def build_send_token_rounds(pairs: list[dict[str, int | str]]) -> list[dict[str, int | str | float]]:
    send_token_rounds = []
    send_order: dict[tuple[int, int], int] = defaultdict(int)
    for pair in pairs:
        if str(pair["stage"]) != "SEND_TOKEN":
            continue
        payload = int(pair["begin_payload"])
        start_cycle = int(pair["begin_cycle"])
        end_cycle = int(pair["end_cycle"])
        rank = int(pair["rank"])
        core_id = int(pair["core_id"])
        lane = (rank, core_id)
        send_token_rounds.append({
            "rank": rank,
            "core_id": core_id,
            "send_order": send_order[lane],
            "to_rank_id": (payload >> 24) & 0xFF,
            "token_id": (payload >> 8) & 0xFFFF,
            "topk_id": payload & 0xFF,
            "tk_index": int(pair["end_payload"]),
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "duration_cycles": int(pair["duration_cycles"]),
            "start_us": start_cycle / CYCLES_PER_US,
            "end_us": end_cycle / CYCLES_PER_US,
            "duration_us": int(pair["duration_cycles"]) / CYCLES_PER_US,
            "format_version": int(pair["format_version"]),
        })
        send_order[lane] += 1
    return send_token_rounds


def align_cycles(rows: list[dict[str, int | str]], pairs: list[dict[str, int | str]], keep_origin: bool) -> None:
    if keep_origin:
        return
    origin: dict[int, int] = {}
    for row in rows:
        rank = int(row["rank"])
        cycle = int(row["cycle"])
        origin[rank] = min(origin.get(rank, cycle), cycle)
    for row in rows:
        row["cycle"] = int(row["cycle"]) - origin[int(row["rank"])]
    for pair in pairs:
        rank_origin = origin[int(pair["rank"])]
        pair["begin_cycle"] = int(pair["begin_cycle"]) - rank_origin
        pair["end_cycle"] = int(pair["end_cycle"]) - rank_origin


def write_csv(path: Path, rows: list[dict[str, int | str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_trace(path: Path, pairs: list[dict[str, int | str]],
                recv_rounds: list[dict[str, int | str | float]],
                recv_slot_arrivals: list[dict[str, int | str | float]], freq_mhz: float) -> None:
    events = []
    for pair in pairs:
        stage = str(pair["stage"])
        payload = int(pair["payload"])
        args = {
            "description": STAGE_DESCRIPTIONS.get(stage, stage),
            "begin_payload": payload,
            "end_payload": int(pair["end_payload"]),
        }
        if stage == "KERNEL":
            args["send_token_cores"] = payload >> 16
            args["cumsum_cores"] = payload & 0xFFFF
        if stage == "SEND_TOKEN":
            args["to_rank_id"] = (payload >> 24) & 0xFF
            args["token_id"] = (payload >> 8) & 0xFFFF
            args["topk_id"] = payload & 0xFF
            args["tk_index"] = int(pair["end_payload"])
        if stage == "RECV_ROUND_BEGIN":
            args["token_local_idx"] = payload >> 16
            args["token_id"] = payload & 0xFFFF
        events.append({
            "name": stage,
            "cat": "FullMeshV2",
            "ph": "X",
            "pid": f"rank{pair['rank']}",
            "tid": f"AIV{pair['core_id']}",
            "ts": float(pair["begin_cycle"]) / freq_mhz,
            "dur": float(pair["duration_cycles"]) / freq_mhz,
            "args": args,
        })
    for recv_round in recv_rounds:
        events.append({
            "name": "RECV_ROUND_BEGIN",
            "cat": "FullMeshV2",
            "ph": "i",
            "s": "t",
            "pid": f"rank{recv_round['rank']}",
            "tid": f"AIV{recv_round['core_id']}",
            "ts": float(recv_round["start_cycle"]) / freq_mhz,
            "args": {
                "ready_order": recv_round["ready_order"],
                "round_id": recv_round["round_id"],
                "token_id": recv_round["token_id"],
                "gap_cycles": recv_round["gap_cycles"],
            },
        })
    for arrival in recv_slot_arrivals:
        events.append({
            "name": "RECV_SLOT_ARRIVE",
            "cat": "FullMeshV2",
            "ph": "i",
            "s": "t",
            "pid": f"rank{arrival['rank']}",
            "tid": f"AIV{arrival['core_id']}",
            "ts": float(arrival["observed_cycle"]) / freq_mhz,
            "args": {
                "rank_arrival_order": arrival["rank_arrival_order"],
                "core_arrival_order": arrival["core_arrival_order"],
                "token_id": arrival["token_id"],
                "slot_begin": arrival["slot_begin"],
                "slot_end": arrival["slot_end"],
                "slot_count": arrival["slot_count"],
                "core_gap_cycles": arrival["core_gap_cycles"],
            },
        })
    path.write_text(json.dumps({"traceEvents": events}, indent=2), encoding="utf-8")


def write_svg(path: Path, pairs: list[dict[str, int | str]],
              recv_rounds: list[dict[str, int | str | float]], freq_mhz: float) -> None:
    lanes = sorted({(int(pair["rank"]), int(pair["core_id"])) for pair in pairs})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    left = 130
    legend_columns = 2
    legend_rows = (len(TIMELINE_STAGES) + legend_columns - 1) // legend_columns
    top = 43 + legend_rows * 22
    lane_height = 24
    plot_width = 1400
    width = left + plot_width + 30
    height = top + max(1, len(lanes)) * lane_height + 55
    max_cycle = max((int(pair["end_cycle"]) for pair in pairs), default=1)
    scale = plot_width / max(1, max_cycle)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.lane{fill:#fafafa;stroke:#eeeeee}.label{fill:#263238}.title{font-size:16px;font-weight:bold}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="title" x="12" y="22">FullMeshV2 per-core timeline</text>',
    ]
    legend_column_width = (width - 24) / legend_columns
    for index, stage in enumerate(TIMELINE_STAGES):
        row = index // legend_columns
        column = index % legend_columns
        x = 12 + column * legend_column_width
        y = 35 + row * 22
        label = html.escape(f"{stage}: {STAGE_DESCRIPTIONS[stage]}")
        svg.append(f'<rect x="{x:.2f}" y="{y}" width="14" height="14" fill="{COLORS[stage]}"/>')
        svg.append(f'<text class="label" x="{x + 20:.2f}" y="{y + 12}">{label}</text>')
    for lane, index in lane_index.items():
        y = top + index * lane_height
        svg.append(f'<rect class="lane" x="{left}" y="{y}" width="{plot_width}" height="{lane_height}"/>')
        svg.append(f'<text class="label" x="8" y="{y + 16}">rank{lane[0]}/AIV{lane[1]}</text>')
    # Draw wider outer stages first so nested wait/format stages remain visible.
    for pair in sorted(pairs, key=lambda item: int(item["duration_cycles"]), reverse=True):
        stage = str(pair["stage"])
        if stage == "KERNEL":
            continue
        y = top + lane_index[(int(pair["rank"]), int(pair["core_id"]))] * lane_height + 3
        x = left + int(pair["begin_cycle"]) * scale
        rect_width = max(1.0, int(pair["duration_cycles"]) * scale)
        color = COLORS.get(stage, "#607d8b")
        duration_us = int(pair["duration_cycles"]) / freq_mhz
        tooltip = html.escape(
            f"{stage}: {duration_us:.3f} us, begin_payload={pair['begin_payload']}, "
            f"end_payload={pair['end_payload']}")
        svg.append(f'<rect x="{x:.2f}" y="{y}" width="{rect_width:.2f}" height="18" fill="{color}"><title>{tooltip}</title></rect>')
    for recv_round in recv_rounds:
        lane = (int(recv_round["rank"]), int(recv_round["core_id"]))
        if lane not in lane_index:
            continue
        x = left + int(recv_round["start_cycle"]) * scale
        y = top + lane_index[lane] * lane_height + 12
        tooltip = html.escape(
            f"RECV_ROUND_BEGIN: ready_order={recv_round['ready_order']}, "
            f"round_id={recv_round['round_id']}, token_id={recv_round['token_id']}, "
            f"start={float(recv_round['start_us']):.3f} us, "
            f"gap={float(recv_round['gap_us']):.3f} us")
        svg.append(f'<circle cx="{x:.2f}" cy="{y}" r="3.5" fill="{COLORS["RECV_ROUND_BEGIN"]}"><title>{tooltip}</title></circle>')
    total_us = max_cycle / freq_mhz
    svg.append(f'<text x="{left}" y="{height - 18}">0 us</text>')
    svg.append(f'<text x="{left + plot_width - 90}" y="{height - 18}">{total_us:.3f} us</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")




# region Dispatch stage/metric summaries and before-after comparison
DISPATCH_METRIC_FIELDS = {
    0x0172: "send_position_cycles",
    0x0173: "send_input_cycles",
    0x0174: "send_quant_cycles",
    0x0175: "send_buffer_wait_cycles",
    0x0300: "send_metadata_cycles",
    0x0301: "send_pack_cycles",
    0x0302: "send_mte3_issue_cycles",
    0x0303: "count_compute_cycles",
    0x0304: "count_issue_cycles",
    0x0171: "send_tasks",
    0x0305: "send_input_count",
    0x0306: "send_quant_count",
}
DISPATCH_DETAIL_FIELDS = ["rank", "core_id", "role", "send_cores", "cumsum_cores", "stage",
                          "duration_cycles", "duration_us", "end_cycle", "begin_payload", "end_payload"]


def write_dispatch_details(output_dir, rows, pairs):
    kernels = {(int(p["rank"]), int(p["core_id"])): int(p["begin_payload"])
               for p in pairs if p["stage"] == "KERNEL"}
    metric_rows = {}
    for lane, payload in kernels.items():
        send_cores, cumsum_cores = payload >> 16, payload & 0xFFFF
        metric_rows[lane] = dict(rank=lane[0], core_id=lane[1],
                                 role="send_token" if lane[1] < send_cores else "cumsum",
                                 send_cores=send_cores, cumsum_cores=cumsum_cores)
    new_lanes = {(int(p["rank"]), int(p["core_id"])) for p in pairs
                 if p["stage"] == "KERNEL" and int(p["format_version"]) >= 4}
    for lane in new_lanes:
        metric_rows[lane].update({field: 0 for field in DISPATCH_METRIC_FIELDS.values()})
    for row in rows:
        field = DISPATCH_METRIC_FIELDS.get(int(row["event_id"]))
        if field is not None:
            lane = (int(row["rank"]), int(row["core_id"]))
            value = int(row["payload"])
            if field.endswith("_cycles") and value == 0xFFFFFFFF:
                raise ValueError(f"{lane}: accumulated metric {field} saturated; recollect shorter work")
            metric_rows.setdefault(lane, dict(rank=lane[0], core_id=lane[1]))[field] = value
    details = []
    for pair in pairs:
        lane = (int(pair["rank"]), int(pair["core_id"]))
        meta = metric_rows.get(lane, {})
        details.append(dict(rank=lane[0], core_id=lane[1], role=meta.get("role", "unknown"),
                            send_cores=meta.get("send_cores", ""), cumsum_cores=meta.get("cumsum_cores", ""),
                            stage=pair["stage"], duration_cycles=int(pair["duration_cycles"]),
                            duration_us=int(pair["duration_cycles"]) / CYCLES_PER_US,
                            end_cycle=pair["end_cycle"], begin_payload=pair["begin_payload"],
                            end_payload=pair["end_payload"]))
    write_csv(output_dir / "fullmesh_v2_dispatch_detail.csv", details, DISPATCH_DETAIL_FIELDS)
    write_csv(output_dir / "fullmesh_v2_dispatch_metrics.csv", list(metric_rows.values()),
              ["rank", "core_id", "role", "send_cores", "cumsum_cores"] + list(DISPATCH_METRIC_FIELDS.values()))
    return details


def compare_dispatch_results(baseline_dir, current_dir):
    # Group by role: core IDs may change role when the send/cumsum split changes.
    def load_grouped(directory, filename, metric_table):
        grouped = defaultdict(list)
        with (Path(directory) / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if metric_table:
                    for field in DISPATCH_METRIC_FIELDS.values():
                        if row.get(field, "") != "":
                            grouped[(row["role"], field)].append(int(row[field]))
                else:
                    grouped[(row["role"], row["stage"])].append(int(row["duration_cycles"]))
        return grouped
    results = []
    for kind, filename in [("stage", "fullmesh_v2_dispatch_detail.csv"),
                           ("metric", "fullmesh_v2_dispatch_metrics.csv")]:
        before = load_grouped(baseline_dir, filename, kind == "metric")
        after = load_grouped(current_dir, filename, kind == "metric")
        for role, stage in sorted(set(before) | set(after)):
            b, a = before.get((role, stage), []), after.get((role, stage), [])
            row = dict(kind=kind, role=role, stage=stage, before_samples=len(b), after_samples=len(a))
            for label, values in [("before", b), ("after", a)]:
                row[label + "_mean"] = sum(values) / len(values) if values else ""
                row[label + "_max"] = max(values) if values else ""
            row["delta_mean"] = row["after_mean"] - row["before_mean"] if a and b else ""
            row["delta_max"] = row["after_max"] - row["before_max"] if a and b else ""
            row["change"] = ("missing" if not a or not b else
                             "slower" if row["delta_max"] > 0 else
                             "faster" if row["delta_max"] < 0 else "equal")
            if kind == "metric" and not stage.endswith("_cycles"):
                row["change"] = "workload_count"
            results.append(row)
    write_csv(Path(current_dir) / "fullmesh_v2_comparison.csv", results,
              ["kind", "role", "stage", "before_samples", "after_samples", "before_mean", "after_mean",
               "delta_mean", "before_max", "after_max", "delta_max", "change"])
    print("[PROFILE] comparison delta=after-before; cycles except *_count/send_tasks. "
          "Stage maxima can come from different cores; nested stages must not be added.")

# endregion


def parse_profiler_files(paths, output_dir: Path):
    rows = []
    for index, file_path in enumerate(sorted(Path(path) for path in paths)):
        rows.extend(parse_binary(file_path, rank_from_path(file_path, index)))
    if not rows:
        raise RuntimeError("no profiler events found")

    pairs = pair_events(rows)
    lanes = {(int(row["rank"]), int(row["core_id"])) for row in rows}
    kernels = [(int(pair["rank"]), int(pair["core_id"])) for pair in pairs if pair["stage"] == "KERNEL"]
    if set(kernels) != lanes or len(kernels) != len(lanes):
        raise ValueError("each recorded core must have exactly one complete KERNEL interval")
    align_cycles(rows, pairs, False)
    summaries = build_core_summary(pairs)
    send_summary = build_send_summary(rows, pairs)
    recv_rounds = build_recv_rounds(rows)
    recv_token_rounds = build_recv_token_rounds(pairs)
    recv_slot_arrivals = build_recv_slot_arrivals(rows)
    send_token_rounds = build_send_token_rounds(pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dispatch_details(output_dir, rows, pairs)
    write_csv(output_dir / "fullmesh_v2_events.csv", rows,
              ["rank", "core_id", "index", "event_id", "event_name", "payload", "cycle",
               "format_version", "layout"])
    write_csv(output_dir / "fullmesh_v2_stages.csv", pairs,
              ["rank", "core_id", "stage", "begin_cycle", "end_cycle", "duration_cycles", "payload",
               "begin_payload", "end_payload", "format_version"])
    write_csv(output_dir / "fullmesh_v2_core_summary.csv", summaries,
              ["rank", "core_id", "format_version", "role", "kernel_cycles", "kernel_end_cycle",
               "wait_recv_cycles", "wait_recv_polls", "get_cumsum_cycles", "get_cumsum_polls",
               "publish_end_cycle", "drain_end_cycle", "wait_cumsum_cycles", "wait_cumsum_polls",
               "wait_cumsum_end_cycle", "assigned_status", "valid_status", "assigned_tokens",
               "copy_batches", "format_cycles", "format_end_cycle", "local_window_cycles",
               "local_window_end_cycle", "local_tail_cycles"])
    write_csv(output_dir / "fullmesh_v2_send_summary.csv", send_summary, list(SEND_SUMMARY_FIELDS))
    write_csv(output_dir / "fullmesh_v2_send_token_rounds.csv", send_token_rounds, list(SEND_TOKEN_FIELDS))
    write_csv(output_dir / "fullmesh_v2_recv_rounds.csv", recv_rounds, list(RECV_ROUND_FIELDS))
    write_csv(output_dir / "fullmesh_v2_recv_token_rounds.csv", recv_token_rounds, list(RECV_TOKEN_FIELDS))
    write_csv(output_dir / "fullmesh_v2_recv_slot_arrivals.csv", recv_slot_arrivals,
              list(RECV_SLOT_ARRIVE_FIELDS))
    write_trace(output_dir / "fullmesh_v2_trace.json", pairs, recv_rounds, recv_slot_arrivals, CYCLES_PER_US)
    write_svg(output_dir / "fullmesh_v2_timeline.svg", pairs, recv_rounds, CYCLES_PER_US)
    print(f"parsed {len(rows)} events, {len(pairs)} stage intervals, "
          f"{len(send_token_rounds)} send token intervals, {len(recv_rounds)} receive rounds and "
          f"{len(recv_token_rounds)} receive token intervals and "
          f"{len(recv_slot_arrivals)} receive slot arrivals "
          f"into {output_dir}")
    print(f"cycles_per_us={CYCLES_PER_US:g}")
    return rows, pairs


def parse_fullmesh_v2_profiler():
    if comm_alg != "fullmesh_v2":
        return

    rank_begin = server_index * rank_per_dev
    paths = [Path(f"fullmesh_v2_prof_rank{rank}.bin")
             for rank in range(rank_begin, rank_begin + rank_per_dev)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing rank profiler dumps: {missing}")
    parse_profiler_files(paths, Path("dispatch_profiler_result"))


if __name__ == "__main__":
    # region 测试前准备工作

    if not is_dispatch or is_combine or is_continue or is_cpu_only:
        raise RuntimeError("dispatch profiler requires is_dispatch=1, is_combine=is_continue=0, is_cpu_only=False")
    if dispatch_comm_alg != "fullmesh_v2":
        raise RuntimeError("dispatch profiler requires comm_alg=fullmesh_v2 (A5 instrumented path)")
    # Never parse another run's dump when a child rank fails.
    for profile_rank in range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev):
        Path(f"fullmesh_v2_prof_rank{profile_rank}.bin").unlink(missing_ok=True)
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
    slow_rank = generate_slow_rank(random_slow_rank_num, ep_world_size, specified_slow_rank)

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
            x_active_mask_list[0][i * bs:(i + 1) * bs] = False
    x_active_mask_list_ela_ori = deepcopy(x_active_mask_list)
    if is_zero_compute_expert:
        if not is_2d_mask:
            is_2d_mask = 1
            x_active_mask_list = [original_tensor.unsqueeze(1).repeat(1, k) for original_tensor in x_active_mask_list]
        for x_active_mask, expert_ids in zip(x_active_mask_list, expert_ids_list):
            x_active_mask[expert_ids >= moe_expert_num] = False

    const_expert_alpha_1, const_expert_alpha_2, const_expert_v = gen_const_expert_alpha_1(), gen_const_expert_alpha_2(), gen_const_expert_v()

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
    # 生成各ep域的expert_scales。mode=1/2时算子侧不使用权重，combine golden按全1处理。
    expert_scales_list = gen_expert_scales()
    combine_golden_expert_scales_list = (
        expert_scales_list if combine_expertscale == 0 else gen_expert_scales(fill_value=1.0)
    )
    combine_input_expert_scales_list = expert_scales_list

    # region 生成dispatch算子的golden值

    (
        golden_expand_x_list, golden_expand_idx_list, golden_expand_idx_masked_list, golden_tokens_list,
        golden_ep_recv_count_list, golden_dynamic_scales_list, middle_idx_list, golden_triplets_expand_idx_list,
        golden_expand_scales_list
    ) = gen_dispatch_golden(
        local_moe_expert_num_list,
        x_list, expert_ids_list, expert_scales_list, x_active_mask_list, x_active_mask_list_ela_ori,
        padded_scales_list, table1_elastic, available_ranks, table2_elastic, elastic_info
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
        golden_expand_scales = [
            golden_expand_scales_list[tp_id][ep_id]
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
        golden_expand_x, golden_ep_recv_count, golden_tokens, golden_dynamic_scales, golden_expand_idx, golden_expand_idx_masked, golden_triplets_expand_idx, golden_expand_scales = (
            golden_expand_x_list[0],
            golden_ep_recv_count_list[0],
            golden_tokens_list[0],
            golden_dynamic_scales_list[0],
            golden_expand_idx_list[0],
            golden_expand_idx_masked_list[0],
            golden_triplets_expand_idx_list[0],
            golden_expand_scales_list[0]
        )
        expert_num_list = [1] * shared_expert_rank_num + local_moe_expert_num_list[0].tolist()
        golden_tokens = split_tensor(golden_tokens, expert_num_list)
        golden_tp_recv_count = [torch.zeros(size=(tp_world_size,), dtype=torch.int32) for _ in range(world_size)]

    if expert_token_nums_type == 0:
        for i in range(len(golden_tokens)):
            golden_tokens[i] = torch.cumsum(golden_tokens[i], dim=0)

    # endregion

    # region 生成combine算子golden值

    if is_quant or is_noq_special:
        for tp_id in range(tp_world_size):
            golden_expand_x_list[tp_id] = [convert_expand_x_dtype(tensor) for tensor in golden_expand_x_list[tp_id]]

    if tp_world_size == 2:
        middle_shared_x_list, middle_moe_x_list = gen_reduce_scatter_golden(golden_expand_x_list, golden_tp_recv_count)
    else:
        expand_x = golden_expand_x_list[0]
        middle_shared_x_list, middle_moe_x_list = (
            [expand_x[:shared_expert_rank_num_ela]],
            [torch.cat(expand_x[shared_expert_rank_num_ela:])],
        )

    golden_x = gen_combine_golden(
        middle_shared_x_list, middle_moe_x_list, middle_idx_list, combine_golden_expert_scales_list,
        golden_tokens_list, x_active_mask_list, x_active_mask_list_ela_ori, shared_expert_x_list,
        x_list, expert_ids_list, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
    )
    golden_x_ori = deepcopy(golden_x)

    if is_elastic:
        no_elastic_shared_ranks_list = list(range(len(available_ranks)))
        if is_quant or is_noq_forward:
            if shared_expert_rank_num != shared_expert_rank_num_ela:
                golden_dynamic_scales = golden_dynamic_scales[:elastic_shared_rank] + golden_dynamic_scales[shared_expert_rank_num:] + [None] * elastic_shared_ranks
            src_dynamic_scales = [golden_dynamic_scales[i].clone() for i in no_elastic_shared_ranks_list]
        if shared_expert_rank_num != shared_expert_rank_num_ela:
            golden_tokens = golden_tokens[:elastic_shared_rank] + golden_tokens[shared_expert_rank_num:] + [None] * elastic_shared_ranks
            golden_expand_idx = golden_expand_idx[:elastic_shared_rank] + golden_expand_idx[shared_expert_rank_num:] + [None] * elastic_shared_ranks
            golden_expand_scales = golden_expand_scales[:elastic_shared_rank] + golden_expand_scales[shared_expert_rank_num:] + [None] * elastic_shared_ranks
            golden_expand_x = golden_expand_x[:elastic_shared_rank] + golden_expand_x[shared_expert_rank_num:] + [None] * elastic_shared_ranks
            golden_ep_recv_count = golden_ep_recv_count[:elastic_shared_rank] + golden_ep_recv_count[shared_expert_rank_num:] + [None] * elastic_shared_ranks
        src_expand_idx = [golden_expand_idx[i] for i in no_elastic_shared_ranks_list]
        src_expand_scales = [golden_expand_scales[i] for i in no_elastic_shared_ranks_list]
        src_expand_x = [golden_expand_x[i] for i in no_elastic_shared_ranks_list]
        src_ep_recv = [golden_ep_recv_count[i] for i in no_elastic_shared_ranks_list]
        src_tokens = [golden_tokens[i].clone() for i in no_elastic_shared_ranks_list]

        for i in range(len(available_ranks)):
            dst_idx = available_ranks[i]
            if is_quant or is_noq_forward:
                golden_dynamic_scales[dst_idx] = src_dynamic_scales[i]
            golden_ep_recv_count[dst_idx] = src_ep_recv[i]
            golden_expand_idx[dst_idx] = src_expand_idx[i]
            golden_expand_scales[dst_idx] = src_expand_scales[i]
            golden_expand_x[dst_idx] = src_expand_x[i]
            golden_tokens[dst_idx] = src_tokens[i]

        golden_expand_idx = apply_mapping(golden_expand_idx, table1_elastic)
        golden_expand_x_list = [golden_expand_x]
    if is_elastic or elastic_moe_ranks or elastic_shared_ranks or elastic_random_ranks:
        for i, tensor in enumerate(golden_ep_recv_count):
            if tensor is None:
                continue
            if tensor.size(0) < ep_world_size * local_moe_expert_num:
                pad_size = ep_world_size * local_moe_expert_num - tensor.size(0)
                tensor = F.pad(tensor, (0, pad_size), "constant", 0)
                golden_ep_recv_count[i] = tensor

    # exit(0)

    # endregion

    print("Generate golden success.")

    # ========================================
    # CPU模式: 保存golden数据到.pt文件后退出
    # ========================================
    if is_cpu_only:
        def _collect_golden(obj, prefix=""):
            """递归收集golden数据，处理嵌套列表和None"""
            if obj is None:
                return {"__none__": True}
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, list):
                return [_collect_golden(item, f"{prefix}[{i}]") for i, item in enumerate(obj)]
            return obj

        golden_save = {
            "golden_expand_x": _collect_golden(golden_expand_x),
            "golden_expand_idx": _collect_golden(golden_expand_idx),
            "golden_expand_idx_masked": _collect_golden(golden_expand_idx_masked),
            "golden_tokens": _collect_golden(golden_tokens),
            "golden_ep_recv_count": _collect_golden(golden_ep_recv_count),
            "golden_dynamic_scales": _collect_golden(golden_dynamic_scales),
            "golden_expand_scales": _collect_golden(golden_expand_scales),
            "golden_triplets_expand_idx": _collect_golden(golden_triplets_expand_idx),
            "golden_tp_recv_count": _collect_golden(golden_tp_recv_count),
            "golden_x": _collect_golden(golden_x),
            "golden_x_ori": _collect_golden(golden_x_ori),
            "golden_expand_x_list": _collect_golden(golden_expand_x_list),
            "middle_idx_list": _collect_golden(middle_idx_list),
            "expert_scales_list": _collect_golden(expert_scales_list),
            "combine_golden_expert_scales_list": _collect_golden(combine_golden_expert_scales_list),
        }
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_original.pt")
        torch.save(golden_save, save_path)
        print(f"Golden data saved to {save_path}")
        exit(0)

    # endregion

    server_index_el = server_index if not is_elastic or (server_index < server_elastic_index) else (server_index - 1)
    server_ranks = work_server_ranks(server_index)
    server_ranks_dis = work_server_ranks(server_index_el)
    if is_dispatch:
        [
            npu_expand_x, npu_dynamic_scales, npu_expand_idx,
            npu_tokens, npu_eprecv_count, npu_tprecv_count, npu_expand_scales
        ] = gen_npu(
            run_dispatch_npu, available_ranks,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in (
                x_active_mask_list if not (is_elastic or is_zero_compute_expert) else x_active_mask_list_ori)],
            scales_list=scales_list,
            expert_scales_list=[chunk_tensor(expert_scales, ep_world_size) for expert_scales in expert_scales_list],
            elastic_info=elastic_info,
            slow_rank=slow_rank,
        )
        parse_fullmesh_v2_profiler()
        print("[PROFILE] expand_x contains debug records; dispatch precision comparison is skipped.")
        sys.exit(0)
        golden_tokens_el = deepcopy(golden_tokens)
        golden_ep_recv_count_el = deepcopy(golden_ep_recv_count)
        golden_expand_x_el = deepcopy(golden_expand_x)
        golden_dynamic_scales_el = deepcopy(golden_dynamic_scales)
        golden_expand_idx_el = deepcopy(golden_expand_idx)
        golden_expand_scales_el = deepcopy(golden_expand_scales)
        if is_elastic or elastic_moe_ranks or elastic_shared_ranks or elastic_random_ranks:
            golden_tokens_el = [golden_tokens_el[i] for i in available_ranks]
            golden_ep_recv_count_el = [golden_ep_recv_count_el[i] for i in available_ranks]
            golden_expand_x_el = [golden_expand_x_el[i] for i in available_ranks]
            golden_expand_scales_el = [golden_expand_scales_el[i] for i in available_ranks]
            if is_quant or is_noq_forward:
                golden_dynamic_scales_el = [golden_dynamic_scales_el[i] for i in available_ranks]
            golden_expand_idx_el = [golden_expand_idx_el[i] for i in available_ranks]
            npu_tokens = [npu_tokens[i] for i in available_ranks] if server_num == 1 else npu_tokens
            npu_eprecv_count = [npu_eprecv_count[i] for i in available_ranks] if server_num == 1 else npu_eprecv_count
            npu_expand_x = [npu_expand_x[i] for i in available_ranks] if server_num == 1 else npu_expand_x
            npu_expand_scales = [npu_expand_scales[i] for i in available_ranks] if server_num == 1 else npu_expand_scales
            if is_quant or is_noq_forward:
                npu_dynamic_scales = [npu_dynamic_scales[i] for i in
                                      available_ranks] if server_num == 1 else npu_dynamic_scales
            npu_expand_idx = [npu_expand_idx[i] for i in available_ranks] if server_num == 1 else npu_expand_idx
            for idx in range(shared_expert_rank_num_ela):
                golden_ep_recv_count_el[idx] = golden_ep_recv_count_el[idx][:len(available_ranks)]
                if server_num == 1:
                    npu_eprecv_count[idx] = npu_eprecv_count[idx][:len(available_ranks)]
                    npu_tokens[idx] = npu_tokens[idx][0:1]

            for idx in range(shared_expert_rank_num_ela, len(available_ranks)):
                golden_ep_recv_count_el[idx] = golden_ep_recv_count_el[idx][
                                               :len(available_ranks) * local_moe_expert_num]
                if server_num == 1:
                    npu_eprecv_count[idx] = npu_eprecv_count[idx][:len(available_ranks) * local_moe_expert_num]
            if server_num > 1:
                if server_index_el < shared_expert_rank_num_ela // rank_per_dev:
                    npu_eprecv_count = [tensor_recv[:len(available_ranks)] for tensor_recv in npu_eprecv_count]
                    npu_tokens = [t[0:1] for t in npu_tokens]
                elif server_index_el == shared_expert_rank_num // rank_per_dev:
                    for idx in range(shared_expert_rank_num % rank_per_dev):
                        npu_eprecv_count[idx] = npu_eprecv_count[idx][:len(available_ranks)]
                        npu_tokens[idx] = npu_tokens[idx][0:1]
                    for idx in range(shared_expert_rank_num % rank_per_dev, rank_per_dev):
                        npu_eprecv_count[idx] = npu_eprecv_count[idx][:len(available_ranks) * local_moe_expert_num]
                else:
                    npu_eprecv_count = [tensor_recv[:len(available_ranks) * local_moe_expert_num] for tensor_recv in
                                        npu_eprecv_count]
        print("Generate dispatch npu success.")

        if is_performance and (graph_type == 0 or is_noq_forward):
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
                "int32",
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
                    "int32",
                    False,
                    available_ranks,
                )
                result_dispatch.update(res)

            # 比较 expand x
            res = compare_output(
                ComparedOutput.EXPAND_X,
                golden_expand_x_el[server_ranks_dis],
                npu_expand_x,
                expandx_dtype,
                True,
                available_ranks,
            )
            result_dispatch.update(res)

            # expertScales有效时比较expand_scales；None/空Tensor场景下该输出无效。
            if dispatch_expertscale == 0:
                def is_shared_expand_scales_rank(rank_id, local_rank_id):
                    if shared_expert_rank_num == 0:
                        return False
                    if is_elastic and server_num == 1:
                        return local_rank_id < shared_expert_rank_num_ela
                    return rank_id // tp_world_size < shared_expert_rank_num

                res = compare_output(
                    ComparedOutput.EXPAND_SCALES,
                    golden_expand_scales_el[server_ranks_dis],
                    npu_expand_scales,
                    "float32",
                    True,
                    available_ranks,
                    is_shared_expand_scales_rank,
                )
                result_dispatch.update(res)

            # 比较 dynamic scales
            if (is_quant and quant_mode != 1) or is_noq_forward:
                if (quant_mode in (4, 5) or (is_noq_forward and dtype_map.get(expandx_dtype, "").startswith("float4"))) and npu_dynamic_scales[0].dtype != torch.uint8:
                    npu_dynamic_scales = [npu_dynamic_scales[i].view(torch.uint8) for i in range(len(npu_dynamic_scales))]
                # fp4场景的scales类型为fp8_e8m0，其余为float32
                # scales_compare_dtype = "float8_e8m0" if (is_noq_forward and dtype_map.get(expandx_dtype, "").startswith("float4")) else "float32"
                res = compare_output(
                    ComparedOutput.DYNAMIC_SCALES,
                    golden_dynamic_scales[server_ranks],
                    npu_dynamic_scales,
                    "float32",
                    False,
                    available_ranks,
                )
                result_dispatch.update(res)

            # 比较 expand idx
            res = compare_output(
                ComparedOutput.EXPAND_IDX,
                golden_expand_idx_el[server_ranks_dis] if is_ccu else golden_triplets_expand_idx[server_ranks],
                npu_expand_idx,
                "int32",
                True,
                available_ranks,
            )
            result_dispatch.update(res)

            # 比较 expert token nums
            res = compare_output(
                ComparedOutput.EXPERT_TOKEN_NUMS,
                golden_tokens_el[server_ranks_dis],
                npu_tokens,
                "int64",
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
            expand_idx_list=golden_expand_idx if is_ccu else golden_triplets_expand_idx,
            ep_send_count_list=golden_ep_recv_count,
            tp_send_count_list=golden_tp_recv_count,
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in (
                x_active_mask_list if not (is_elastic or is_zero_compute_expert) else x_active_mask_list_ori)],
            expert_scales_list=(
                [chunk_tensor(expert_scales, ep_world_size) for expert_scales in combine_input_expert_scales_list]
                if combine_input_expert_scales_list is not None else None
            ),
            shared_expert_x_list=shared_expert_x_list,
            elastic_info=elastic_info,
            ori_x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            slow_rank=slow_rank,
        )

        print("Generate combine npu success.")

        if is_performance and graph_type == 0:
            combine_performance = get_combine_performance()

        parse_fullmesh_v2_profiler()
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
                npu_x = [npu_x_per_card[mask_all_per_card] for npu_x_per_card, mask_all_per_card in zip(npu_x,
                                                                                                        mask_per_card if server_num == 1 else
                                                                                                        mask_per_card_server[work_server_ranks(server_index)])]
                golden_x_rank = [golden_x_per_card[mask_all_per_card] for golden_x_per_card, mask_all_per_card in
                                 zip(golden_x, mask_per_card)]

            res = compare_output(
                ComparedOutput.X,
                golden_x_rank[server_ranks] if is_2d_mask else golden_x[server_ranks],
                npu_x,
                "float32",
                True,
                available_ranks
            )
            result_combine.update(res)

    if is_continue:
        [npu_x] = gen_npu(
            run_continue_npu, available_ranks,
            x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            expert_ids_list=[chunk_tensor(expert_ids, ep_world_size) for expert_ids in expert_ids_list],
            x_active_mask_list=[chunk_tensor(x_active_mask, ep_world_size) for x_active_mask in (
                x_active_mask_list if not (is_elastic or is_zero_compute_expert) else x_active_mask_list_ori)],
            scales_list=scales_list,
            expert_scales_list=(
                [chunk_tensor(expert_scales, ep_world_size) for expert_scales in combine_input_expert_scales_list]
                if combine_input_expert_scales_list is not None else None
            ),
            golden_expand_x_list=golden_expand_x_list,
            shared_expert_x_list=shared_expert_x_list,
            elastic_info=elastic_info,
            ori_x_list=[chunk_tensor(x, ep_world_size) for x in x_list],
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            slow_rank=slow_rank,
        )

        if is_performance:
            continue_performance = get_continue_performance()

        print("Generate continue npu success.")
        parse_fullmesh_v2_profiler()

        print("##################Start Continue Compare##################")
        if is_elastic and server_num > 1 and server_index == server_elastic_index:
            pass
        else:
            result_continue = {}
            # 比较 x
            if is_elastic:
                golden_x = golden_x_ori
                if server_num == 1:
                    npu_x = [npu_x[i] for i in available_ranks]
                    golden_x = [golden_x[i] for i in available_ranks]
            if is_variable_bs and is_zero_compute_expert:
                npu_x = pad_tensor(npu_x)

            if is_2d_mask:
                npu_x = [npu_x_per_card[mask_all_per_card] for npu_x_per_card, mask_all_per_card in zip(npu_x,
                                                                                                        mask_per_card if server_num == 1 else
                                                                                                        mask_per_card_server[
                                                                                                            work_server_ranks(
                                                                                                                server_index)])]
                golden_x_rank = [golden_x_per_card[mask_all_per_card] for golden_x_per_card, mask_all_per_card in
                                 zip(golden_x, mask_per_card)]
            res = compare_output(
                ComparedOutput.X,
                golden_x_rank[server_ranks] if is_2d_mask else golden_x[server_ranks],
                npu_x,
                "float32",
                True,
                available_ranks
            )
            result_continue.update(res)

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

            if is_performance and (graph_type == 0 or is_noq_forward):
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
