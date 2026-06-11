"""
URMA P2P Benchmark Tool (SQE-level HcclCombinOpParam URMA Communication)

Dual-mode design:
- mode=0: Overhead+Latency (Per-operation timing breakdown, 256B~2MB)
- mode=1: Bandwidth (P2P single-link throughput)

Note: This tool uses SQE-level HcclCombinOpParam URMA communication. The operator
declares HCCL communication domain via .MC2().HcclGroup({"group"}), the framework
automatically creates HCCL context, no manual mc2Context required.
Operator invoked via npu_ops_transformer.ops.npu_urma_p2p_benchmark.
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "torch_extensions_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir

import sys

# 将项目torch_extension目录插入sys.path最前面，确保优先加载项目自带的npu_ops_transformer
# （包含urma_p2p_benchmark），而非CANN自带的版本（不含urma_p2p_benchmark）
# 注意: 路径从脚本目录(mc2/tools/urma_p2p_benchmark/)向上3级到项目根目录
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
torch_ext_path = os.path.join(project_root, 'torch_extension')
sys.path.insert(0, torch_ext_path)

import socket
import argparse
import torch
import torch_npu

# 清除可能被torch_npu提前导入的CANN自带npu_ops_transformer缓存
# 必须在torch_npu导入之后执行，因为torch_npu内部可能已经加载了CANN的npu_ops_transformer
for key in list(sys.modules.keys()):
    if key.startswith('npu_ops_transformer'):
        del sys.modules[key]

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
import traceback
import time
import pandas as pd
import numpy as np
from math import ceil
from typing import List, Tuple, Dict, Any

import npu_ops_transformer


# region 命令行参数解析

def parse_args():
    parser = argparse.ArgumentParser(description='URMA P2P基准测试工具')
    parser.add_argument('--test_type', choices=['latency', 'bandwidth', 'all'], default='all',
                        help='Test type: latency/bandwidth/all')
    parser.add_argument('--data_sizes', type=int, nargs='+',
                        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152],
                        help='数据大小列表(字节), 从256B到2MB')
    parser.add_argument('--iterations', type=int, default=100, help='迭代次数(两种模式共用)')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--dst_rank', type=int, default=-1, help='P2P目标rank, -1=自动((rank+1)%world_size)')
    parser.add_argument('--world_size', type=int, default=8, help='通信域rank总数')
    parser.add_argument('--rank_per_dev', type=int, default=8, help='每设备rank数')
    parser.add_argument('--server_num', type=int, default=1, help='服务器数量')
    parser.add_argument('--master_ip', type=str, default='127.0.0.1', help='HCCL初始化主节点IP')
    parser.add_argument('--graph_type', type=int, choices=[0, 3], default=0, help='图模式: 0=单算子, 3=aclgraph')
    parser.add_argument('--output_csv', type=str, default='urma_p2p_benchmark_results.csv', help='CSV输出文件路径')
    parser.add_argument('--output_dir', type=str, default='./urma_p2p_benchmark_output', help='报告输出目录')
    parser.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16', help='数据类型')
    return parser.parse_args()

# endregion


# region 配置参数

def setup_environment(args):
    """根据命令行参数设置全局配置"""
    global graph_type, is_full_core, core_num, aic_core_num, aiv_core_num, aiv_num
    global input_dtype, x_dtype, random_seed
    global server_num, rank_per_dev, world_size, master_ip, server_index, ranks_list, server_ranks
    global profiling_path_base

    # 图模式配置
    graph_type = args.graph_type
    is_full_core = 1
    core_num = 16 if not is_full_core else 32
    aic_core_num = core_num
    aiv_core_num = core_num * 2
    aiv_num = aiv_core_num

    # 数据类型配置
    x_dtype = args.dtype
    input_dtype = torch.bfloat16 if x_dtype == "bf16" else torch.float16
    random_seed = 90

    # 通信域配置
    server_num = args.server_num
    rank_per_dev = args.rank_per_dev
    world_size = args.world_size
    master_ip = args.master_ip
    server_index = 0
    ranks_list = list(range(world_size))
    server_ranks = slice(server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev)

    # Profiling路径基础
    profiling_path_base = './Profiling_Result_urma_p2p_benchmark'

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印配置信息
    print_setting(args)


def print_setting(args):
    """打印当前配置信息"""
    print(f"\n{'=' * 60}")
    print("URMA P2P Benchmark 配置信息")
    print(f"{'=' * 60}")
    print(f"测试类型: {args.test_type}")
    print(f"数据类型: {args.dtype}")
    print(f"服务器数量: {server_num}")
    print(f"每设备rank数: {rank_per_dev}")
    print(f"world_size: {world_size}")
    print(f"图模式: {graph_type}")
    print(f"AIV核数: {aiv_num}")
    print(f"迭代次数: {args.iterations}")
    print(f"预热次数: {args.warmup}")
    print(f"目标rank: {args.dst_rank}")
    print(f"数据大小列表: {args.data_sizes}")
    print(f"输出CSV: {args.output_csv}")
    print(f"输出目录: {args.output_dir}")
    print(f"通信模式: SQE级HcclCombinOpParam URMA")
    print(f"{'=' * 60}\n")

# endregion


# region 数据大小映射函数

def data_size_to_bs_h(data_size_bytes, dtype='fp16', force_bs1=False):
    """将数据字节大小转换为(bs, h)参数对

    Args:
        data_size_bytes: 数据大小(字节)
        dtype: 数据类型('fp16'或'bf16')
        force_bs1: 强制bs=1（用于OverheadLatency模式，只发1个token）

    Returns:
        (bs, h) 参数对, bs为batch size, h为hidden size元素数
    """
    dtype_size = 2  # fp16和bf16都是2字节
    if force_bs1:
        # OverheadLatency模式: 固定bs=1，h=数据大小/元素大小
        # 只发1个token，避免多token顺序Drain导致AIV核超时
        return (1, data_size_bytes // dtype_size)
    mappings = {
        256: (1, 128),       # 256B
        512: (1, 256),       # 512B
        1024: (1, 512),      # 1KB
        2048: (1, 1024),     # 2KB
        4096: (2, 1024),     # 4KB
        8192: (4, 1024),     # 8KB
        16384: (8, 1024),    # 16KB
        32768: (16, 1024),   # 32KB
        65536: (32, 1024),   # 64KB
        131072: (64, 1024),  # 128KB
        262144: (128, 1024), # 256KB
        524288: (256, 1024), # 512KB
        1048576: (512, 1024), # 1MB
        2097152: (1024, 1024), # 2MB
    }
    if data_size_bytes in mappings:
        return mappings[data_size_bytes]
    # 通用计算: bs=1, h=data_size_bytes/dtype_size
    return (1, data_size_bytes // dtype_size)

# endregion


# region HCCL_BUFFSIZE计算

def cal_hccl_buffsize(bs, h, ep_world_size):
    """计算HCCL缓冲区大小(MB)

    Args:
        bs: batch size
        h: hidden size元素数
        ep_world_size: 通信域rank总数

    Returns:
        HCCL_BUFFSIZE字符串值(MB)
    """
    MAX_OUT_DTYPE_SIZE = 2
    WIN_ADDR_ALIGN = 512
    UB_ALIGN = 32
    SCALE_EXPAND_IDX_BUFFER = 44
    DOUBLE_DATA_BUFFER = 2

    token_actual_len = ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1) // UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER
    token_need_size_dispatch = ((token_actual_len + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    actual_size = (bs * token_need_size_dispatch * ep_world_size) * DOUBLE_DATA_BUFFER

    hccl_buffsize = ceil(actual_size / 1024 / 1024 / 2) + 5  # 留出5MB空间余量
    return str(hccl_buffsize)

# endregion


# region 工具函数

CYCLES_PER_US = 50  # 1 us = 50 cycles on Ascend AIV
NS_PER_CYCLE = 20   # 1 cycle = 20ns


def format_data_size(bytes_size):
    """将字节大小格式化为易读字符串"""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size // 1024}KB"
    else:
        return f"{bytes_size // (1024 * 1024)}MB"


def compute_statistics(times_us):
    """计算统计信息

    Args:
        times_us: 时间数组(微秒)

    Returns:
        包含mean/min/max/std/p50/p90/p99的字典
    """
    stats = {
        'mean': np.mean(times_us),
        'min': np.min(times_us),
        'max': np.max(times_us),
        'std': np.std(times_us),
        'p50': np.percentile(times_us, 50),
        'p90': np.percentile(times_us, 90),
        'p99': np.percentile(times_us, 99),
    }
    return stats


def compute_phase_breakdown(perf_info_list):
    """从kernel performance_info列表计算各阶段时延统计

    performance_info为int64 tensor shape (12,)，9个timing slot + 3 padding:
      [0]  PROCESS_START       - Process()开始
      [1]  DUPLICATE_END       - Duplicate(dstRankIds)+PipeBarrier完成
      [2]  INIT_SQ_CQ_LOAD_END - InitRankComm步骤1-6完成（SQ/CQ加载+PICI+CQE初始化+CQ轮询）
      [3]  SQE_TEMPLATE_END    - InitRankComm步骤7-8完成（SQE模板生成+元数据更新）
      [4]  PROCESS_TOKEN_END   - ProcessToken完成（数据写入本地窗口）
      [5]  P2P_SEND_END         - P2pSend完成（SQE填写+写入环形缓冲区）
      [6]  DOORBELL_END        - TriggerDoorBell完成（SendJFSDoorBell完成）
      [7]  POLL_CQ_END         - PollCQCompletion完成（CQ轮询确认）
      [8]  UPDATE_PICI_END     - UpdatePiciCompletion完成（PICI更新）= TOTAL_END

    所有阶段时延统一以us为单位(cycle差 / 50 = us):
      duplicate_us:           [1]-[0] / 50  (PROCESS_START->DUPLICATE_END, us)
      sq_cq_load_us:          [2]-[1] / 50  (DUPLICATE_END->INIT_SQ_CQ_LOAD_END, us)
      sqe_template_us:        [3]-[2] / 50  (INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END, us)
      process_token_us:       [4]-[3] / 50  (SQE_TEMPLATE_END->PROCESS_TOKEN_END, us)
      p2p_send_us:            [5]-[4] / 50  (PROCESS_TOKEN_END->P2P_SEND_END, us)
      trigger_doorbell_us:    [6]-[5] / 50  (P2P_SEND_END->DOORBELL_END, us)
      poll_cq_us:             [7]-[6] / 50  (DOORBELL_END->POLL_CQ_END, us)
      update_pici_us:         [8]-[7] / 50  (POLL_CQ_END->UPDATE_PICI_END, us)
      total_us:               [8]-[0] / 50  (PROCESS_START->UPDATE_PICI_END, us)

    Args:
        perf_info_list: 每次迭代的performance_info numpy数组列表，每个为shape (12,)

    Returns:
        dict，每个phase包含 mean/min/max/std/p50/p90/p99 统计值（us单位）
    """
    # 各阶段名称 -> (end_index, start_index) 映射，统一以us为单位
    phases = {
        'duplicate_us':           (1, 0),
        'sq_cq_load_us':          (2, 1),
        'sqe_template_us':        (3, 2),
        'process_token_us':       (4, 3),
        'p2p_send_us':            (5, 4),
        'trigger_doorbell_us':    (6, 5),
        'poll_cq_us':             (7, 6),
        'update_pici_us':         (8, 7),
        'total_us':               (8, 0),
    }

    result = {}

    for phase_name, (end_idx, start_idx) in phases.items():
        values_us = []
        for perf_info in perf_info_list:
            cycle_end = perf_info[end_idx]
            cycle_start = perf_info[start_idx]
            # For BandwidthTest mode, SLOT[5](DoorBell) may be 0, skip phases involving that slot
            if cycle_end > cycle_start and cycle_start > 0:
                values_us.append((cycle_end - cycle_start) / CYCLES_PER_US)

        if len(values_us) > 0:
            arr = np.array(values_us)
            result[phase_name] = compute_statistics(arr)
        else:
            result[phase_name] = {}

    return result


def print_phase_breakdown(breakdown, title="Phase Breakdown"):
    """打印阶段分解表格

    Args:
        breakdown: compute_phase_breakdown返回的dict
        title: 表格标题
    """
    phase_labels = {
        'duplicate_us':           'PROCESS_START->DUPLICATE_END',
        'sq_cq_load_us':          'DUPLICATE_END->INIT_SQ_CQ_LOAD_END',
        'sqe_template_us':        'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END',
        'process_token_us':       'SQE_TEMPLATE_END->PROCESS_TOKEN_END',
        'p2p_send_us':            'PROCESS_TOKEN_END->P2P_SEND_END',
        'trigger_doorbell_us':    'P2P_SEND_END->DOORBELL_END',
        'poll_cq_us':             'DOORBELL_END->POLL_CQ_END',
        'update_pici_us':         'POLL_CQ_END->UPDATE_PICI_END',
        'total_us':               'PROCESS_START->UPDATE_PICI_END',
    }

    print(f"\n  {title}")
    print(f"  {'Phase':<38} {'Mean':<12} {'Min':<12} {'Max':<12} {'Std':<12} {'P50':<12} {'P90':<12} {'P99':<12}")
    print(f"  {'-' * 122}")

    for phase_name, label in phase_labels.items():
        stats = breakdown.get(phase_name, {})
        if not stats:
            continue
        print(f"  {label:<38} {stats.get('mean', 0):.4f}us    "
              f"{stats.get('min', 0):.4f}us    "
              f"{stats.get('max', 0):.4f}us    "
              f"{stats.get('std', 0):.4f}us    "
              f"{stats.get('p50', 0):.4f}us    "
              f"{stats.get('p90', 0):.4f}us    "
              f"{stats.get('p99', 0):.4f}us")

# endregion


# region SQE级UrmaP2pBenchmark算子调用

def urma_p2p_benchmark_op(x, group, world_size, max_bs, mode,
                          ccl_buffer_size, dst_rank_id_attr, aiv_num):
    """调用UrmaP2pBenchmark算子（SQE级HcclCombinOpParam URMA通信）

    通过npu_ops_transformer.ops.npu_urma_p2p_benchmark调用GE算子。
    HCCL上下文由MC2框架基于.MC2().HcclGroup({"group"})声明自动创建。

    Args:
        x: 输入tensor (bs, h), dtype=float16
        group: HCCL通信域名称
        world_size: 通信域rank数
        max_bs: 最大batch size
        mode: 测试模式 (0=overhead+latency, 1=bandwidth)
        ccl_buffer_size: HCCL缓冲区大小
        dst_rank_id_attr: P2P目标rank属性 (-1=全部rank)
        aiv_num: AIV核数

    Returns:
        (y, receive_cnt, performance_info) 输出tensor三元组
        performance_info: int64 tensor shape (12,), 9个timing slot + 3 padding, kernel内部各阶段cycle计数
        mode=0/1 slot语义:
          [0]=PROCESS_START, [1]=DUPLICATE_END, [2]=INIT_SQ_CQ_LOAD_END,
          [3]=SQE_TEMPLATE_END, [4]=PROCESS_TOKEN_END, [5]=P2P_SEND_END,
          [6]=DOORBELL_END, [7]=POLL_CQ_END, [8]=UPDATE_PICI_END
    """
    y, receive_cnt, performance_info = npu_ops_transformer.ops.npu_urma_p2p_benchmark(
        x, group, world_size, max_bs, mode, ccl_buffer_size, dst_rank_id_attr, aiv_num
    )
    return y, receive_cnt, performance_info

# endregion


# region 性能采集函数

def get_profiling(loop, repeat, warmup, path):
    """创建Profiling配置

    Args:
        loop: 总循环次数
        repeat: 重复次数
        warmup: 预热次数
        path: profiling输出路径

    Returns:
        torch_npu.profiler.profile对象
    """
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
    """删除之前的profiling结果目录"""
    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')


def get_performance_data(profiling_path, op_type='UrmaP2pBenchmark'):
    """从profiling结果提取性能数据（仅用于图模式）

    Args:
        profiling_path: profiling结果路径
        op_type: 算子类型名称

    Returns:
        numpy数组(微秒时间)或None
    """
    execution_times = []
    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                op_info = op_summary[op_summary['OP Type'] == op_type]
                if not op_info.empty:
                    execution_times.extend(op_info['Task Duration(us)'].tolist())

    return np.array(execution_times) if execution_times else None

# endregion


# region 图模式相关

def compile_model(model):
    """编译图模式模型"""
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        sys.exit(1)

    print(f"------------------graph_type: {graph_type}------------------")
    import torchair

    from torchair.configs.compiler_config import CompilerConfig
    if is_full_core:
        compiler_config = None
    else:
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
        print(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")

    if graph_type == 3:
        print('is_aclgraph=1, 测试场景: aclgraph入图模式')
        os.makedirs("./static_kernel", exist_ok=True)
        compiler_config = CompilerConfig()
        print(f"启用aclnn静态内核（aclgraph），当前只支持全核")
        compiler_config.mode = "reduce-overhead"
        # 开启静态Kernel编译
        compiler_config.debug.aclgraph.clone_input = False
        compiler_config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        compiler_config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./static_kernel"

        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    else:
        raise ValueError(f"不支持的 graph_type: {graph_type}")

    return compiled_model


class URMA_P2P_BENCHMARK_GRAPH_Model(torch.nn.Module):
    """URMA P2P Benchmark图模式模型（SQE级HcclCombinOpParam URMA通信）"""

    def __init__(self, loop_count, test_mode, test_max_bs, ccl_buffer_size):
        super().__init__()
        self.loop_count = loop_count
        self.test_mode = test_mode
        self.test_max_bs = test_max_bs
        self.ccl_buffer_size = ccl_buffer_size

    def forward(self, x, group, world_size, max_bs, mode, ccl_buffer_size, dst_rank_id_attr, aiv_num):
        local_loop = self.loop_count
        input_x = x
        output = None
        for _ in range(local_loop):
            output = urma_p2p_benchmark_op(
                x=input_x,
                group=group,
                world_size=world_size,
                max_bs=max_bs,
                mode=mode,
                ccl_buffer_size=ccl_buffer_size,
                dst_rank_id_attr=dst_rank_id_attr,
                aiv_num=aiv_num,
            )
            [y, receive_cnt] = output
            input_x = y[:self.test_max_bs]

        return output

# endregion


# region 通信初始化

def set_device(rank, rank_per_dev_local):
    """设置当前进程使用的NPU设备"""
    torch_npu.npu.set_device(rank % rank_per_dev_local)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank, world_size_local, master_ip_local=None, ranks_list_local=None):
    """初始化HCCL通信域

    Args:
        rank: 当前进程rank
        world_size_local: 通信域总rank数
        master_ip_local: 主节点IP（spawn模式下需传入）
        ranks_list_local: rank列表（spawn模式下需传入）
    Returns:
        (hcomm_info, comm_group) 通信域信息
    """
    if master_ip_local is None:
        master_ip_local = master_ip
    if ranks_list_local is None:
        ranks_list_local = ranks_list
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size_local,
                            init_method=f'tcp://{master_ip_local}:50001')
    print(f"device_{rank} init_process_group success")

    print(f"device {rank} 初始化通信域")
    comm_group = dist.new_group(backend="hccl", ranks=ranks_list_local)

    print(f"[DEBUG] device_{rank} 等待所有进程初始化完成...")
    dist.barrier()
    print(f"[DEBUG] device_{rank} barrier完成")

    torch.npu.synchronize()
    print(f"[DEBUG] device_{rank} NPU同步完成")

    hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    print(f"[DEBUG] device_{rank} HCCL通信域名称: {hcomm_info}, 长度: {len(hcomm_info) if hcomm_info else 0}")

    if not hcomm_info or len(hcomm_info) == 0:
        raise RuntimeError(f"device_{rank} HCCL通信域获取失败")

    dist.barrier()
    torch.npu.synchronize()
    print(f"[DEBUG] device_{rank} 通信域完全就绪")

    return hcomm_info, comm_group

# endregion


# region 输入生成

def generate_p2p_inputs(test_bs, test_data_size, test_world_size):
    """生成P2P测试输入数据

    Args:
        test_bs: batch size
        test_data_size: hidden size元素数
        test_world_size: world size

    Returns:
        x_per_rank 列表
    """
    torch.manual_seed(random_seed)

    x_per_rank = [
        torch.empty([test_bs, test_data_size], dtype=input_dtype).uniform_(-1, 1)
        for _ in range(test_world_size)
    ]

    return x_per_rank

# endregion


# region NPU执行 - SQE级算子调用

def run_urma_p2p_benchmark_npu(queue, rank, x, hcomm_info,
                               test_max_bs=None, test_mode=None, test_world_size=None,
                               dst_rank_id_val=None, is_perf_mode=False, destroy_comm=True,
                               test_loop=None, test_warmup=None, test_profiling_path=None,
                               test_ccl_buffer_size=None, test_rank_per_dev=None,
                               test_master_ip=None, test_graph_type=None,
                               test_aiv_num=None, test_input_dtype=None):
    """执行urma_p2p_benchmark算子（SQE级HcclCombinOpParam URMA通信）

    Args:
        queue: 多进程通信队列
        rank: 当前进程rank
        x: 输入tensor
        hcomm_info: HCCL通信域信息，如果为None则创建新通信域
        test_max_bs: 最大batch size
        test_mode: 测试模式(0=overhead+latency, 1=bandwidth)
        test_world_size: world size
        dst_rank_id_val: P2P目标rank ID（通过属性传递）
        is_perf_mode: 是否为性能测试模式
        destroy_comm: 是否在结束时销毁通信域
        test_loop: 本次测试的循环次数
        test_warmup: 本次测试的预热次数
        test_profiling_path: 本次测试的profiling路径
        test_ccl_buffer_size: HCCL缓冲区大小
        test_rank_per_dev: 每设备rank数（spawn模式下全局变量不可用）
        test_master_ip: HCCL主节点IP（spawn模式下全局变量不可用）
        test_graph_type: 图模式（spawn模式下全局变量不可用）
        test_aiv_num: AIV核数（spawn模式下全局变量不可用）
        test_input_dtype: 输入数据类型（spawn模式下全局变量不可用）
    """
    # spawn模式下子进程不继承全局变量，必须通过参数传递
    global rank_per_dev, master_ip, graph_type, aiv_num, input_dtype
    if test_rank_per_dev is not None: rank_per_dev = test_rank_per_dev
    if test_master_ip is not None: master_ip = test_master_ip
    if test_graph_type is not None: graph_type = test_graph_type
    if test_aiv_num is not None: aiv_num = test_aiv_num
    if test_input_dtype is not None: input_dtype = test_input_dtype

    set_device(rank, rank_per_dev)

    if hcomm_info is None:
        hcomm_info, _ = init_hccl_comm(rank, test_world_size, master_ip, list(range(test_world_size)))

    if test_ccl_buffer_size is None:
        test_ccl_buffer_size = int(cal_hccl_buffsize(test_max_bs, x.size(1), test_world_size))

    print(f'[INFO] device_{rank} 构造urma_p2p_benchmark算子输入数据')

    x_npu = x.to(input_dtype).npu()

    print(f"[DEBUG] device_{rank} x shape: {x_npu.shape}, dtype: {x_npu.dtype}, device: {x_npu.device}")
    print(f"[DEBUG] device_{rank} group: {hcomm_info}, world_size: {test_world_size}, max_bs: {test_max_bs}, mode: {test_mode}, dst_rank_id: {dst_rank_id_val}, ccl_buffer_size: {test_ccl_buffer_size}")

    torch.npu.synchronize()
    print(f"[DEBUG] device_{rank} 开始调用算子, mode={test_mode}")

    # Kernel侧打点计时数据收集
    # performance_info: int64 shape (12,), 9个timing slot + 3 padding
    # [0]=PROCESS_START, [1]=DUPLICATE_END, [2]=INIT_SQ_CQ_LOAD_END,
    # [3]=SQE_TEMPLATE_END, [4]=PROCESS_TOKEN_END, [5]=P2P_SEND_END,
    # [6]=DOORBELL_END, [7]=POLL_CQ_END, [8]=UPDATE_PICI_END
    # CYCLES_PER_US = 50

    performance_info_list = []  # 每次迭代的performance_info数据

    if graph_type == 0:
        if not is_perf_mode:
            y, receive_cnt, perf_info = urma_p2p_benchmark_op(
                x=x_npu,
                group=hcomm_info,
                world_size=test_world_size,
                max_bs=test_max_bs,
                mode=test_mode,
                ccl_buffer_size=test_ccl_buffer_size,
                dst_rank_id_attr=dst_rank_id_val,
                aiv_num=aiv_num,
            )
            torch.npu.synchronize()
            performance_info_list.append(perf_info.cpu().numpy())
        else:
            cur_loop = test_loop if test_loop is not None else 100
            cur_warmup = test_warmup if test_warmup is not None else 10
            repeat = 1
            for i in range(cur_warmup + cur_loop):
                torch.npu.synchronize()
                y, receive_cnt, perf_info = urma_p2p_benchmark_op(
                    x=x_npu,
                    group=hcomm_info,
                    world_size=test_world_size,
                    max_bs=test_max_bs,
                    mode=test_mode,
                    ccl_buffer_size=test_ccl_buffer_size,
                    dst_rank_id_attr=dst_rank_id_val,
                    aiv_num=aiv_num,
                )
                torch.npu.synchronize()
                # warmup迭代不纳入统计数据，只记录正式测试迭代
                if i >= cur_warmup:
                    performance_info_list.append(perf_info.cpu().numpy())
    else:
        # 图模式暂不支持kernel打点，保留原有msprof方式
        torch._dynamo.reset()
        cur_loop = test_loop if test_loop is not None else 100
        cur_warmup = test_warmup if test_warmup is not None else 10
        cur_profiling_path = test_profiling_path if test_profiling_path is not None else profiling_path_base
        tensor = torch.ones(10000, 10000).to(torch.float16).npu()
        model = compile_model(URMA_P2P_BENCHMARK_GRAPH_Model(cur_loop, test_mode, test_max_bs, test_ccl_buffer_size))
        loop_time = 2
        repeat_time = 1
        warm_time = 1
        with get_profiling(loop_time, repeat_time, warm_time, cur_profiling_path) as prof:
            for i in range(loop_time):
                y, receive_cnt, performance_info = model(
                    x=x_npu,
                    group=hcomm_info,
                    world_size=test_world_size,
                    max_bs=test_max_bs,
                    mode=test_mode,
                    ccl_buffer_size=test_ccl_buffer_size,
                    dst_rank_id_attr=dst_rank_id_val,
                    aiv_num=aiv_num,
                )
                if i == 1:
                    for _ in range(100):
                        torch.exp(tensor)
                prof.step()

    torch.npu.synchronize()
    torch.npu.empty_cache()

    if destroy_comm:
        dist.destroy_process_group()
        print(f'rank {rank} destroyed process group')

    print(f'rank {rank} npu finished! \n')

    queue.put([
        rank,
        [
            y.cpu(), receive_cnt.cpu(), performance_info_list
        ]
    ])

# endregion


# region 多配置共享通信域执行

def run_urma_p2p_benchmark_npu_multi_config(queue, rank, configs,
                                              test_rank_per_dev=None,
                                              test_master_ip=None,
                                              test_graph_type=None,
                                              test_aiv_num=None,
                                              test_input_dtype=None):
    """多配置带宽测试，共享通信域，使用kernel timing替代msprof

    Args:
        queue: 多进程通信队列
        rank: 当前进程rank
        configs: 配置列表，每个元素为:
            (x, test_max_bs, test_data_size_bytes, test_world_size,
             config_index, dst_rank_id_val, test_loop, test_warmup, ccl_buffer_size) 元组
        test_rank_per_dev: 每设备rank数（spawn模式下全局变量不可用）
        test_master_ip: HCCL主节点IP（spawn模式下全局变量不可用）
        test_graph_type: 图模式（spawn模式下全局变量不可用）
        test_aiv_num: AIV核数（spawn模式下全局变量不可用）
        test_input_dtype: 输入数据类型（spawn模式下全局变量不可用）
    """
    # spawn模式下子进程不继承全局变量，必须通过参数传递
    global rank_per_dev, master_ip, graph_type, aiv_num, input_dtype
    if test_rank_per_dev is not None: rank_per_dev = test_rank_per_dev
    if test_master_ip is not None: master_ip = test_master_ip
    if test_graph_type is not None: graph_type = test_graph_type
    if test_aiv_num is not None: aiv_num = test_aiv_num
    if test_input_dtype is not None: input_dtype = test_input_dtype

    set_device(rank, rank_per_dev)

    if not configs:
        queue.put([rank, {}])
        return

    _, _, _, test_world_size, _, _, _, _, _ = configs[0]

    print(f'[INFO] device_{rank} 创建通信域（多配置共享）')
    hcomm_info, _ = init_hccl_comm(rank, test_world_size, master_ip, list(range(test_world_size)))

    results = {}  # config_index -> performance_info_list

    for i, (x, test_max_bs, test_data_size_bytes, test_world_size,
            config_index, dst_rank_id_val, test_loop, test_warmup, ccl_buffer_size) in enumerate(configs):
        test_bs = x.size(0)
        print(f"\n{'=' * 60}")
        print(f"[Rank {rank}] Config [{config_index}] bs={test_bs}, data_size={test_data_size_bytes}, "
              f"max_bs={test_max_bs}, dst_rank={dst_rank_id_val}, mode=1(带宽)")
        print(f"{'=' * 60}")

        print(f'[INFO] device_{rank} 构造urma_p2p_benchmark算子输入数据')

        x_npu = x.to(input_dtype).npu()

        torch.npu.synchronize()
        print(f"[DEBUG] device_{rank} 开始调用算子")
        dist.barrier()

        # 使用kernel timing收集性能数据（替代msprof）
        config_perf_info_list = []
        cur_warmup = test_warmup if test_warmup is not None else 10

        if graph_type == 0:
            # 单算子模式：多次调用算子，每次收集performance_info
            for j in range(cur_warmup + test_loop):
                torch.npu.synchronize()
                y, receive_cnt, perf_info = urma_p2p_benchmark_op(
                    x=x_npu,
                    group=hcomm_info,
                    world_size=test_world_size,
                    max_bs=test_max_bs,
                    mode=1,  # bandwidth mode
                    ccl_buffer_size=ccl_buffer_size,
                    dst_rank_id_attr=dst_rank_id_val,
                    aiv_num=aiv_num,
                )
                torch.npu.synchronize()
                # warmup迭代不纳入统计数据
                if j >= cur_warmup:
                    config_perf_info_list.append(perf_info.cpu().numpy())
        else:
            # 图模式暂不支持kernel打点，保留原有msprof方式
            torch._dynamo.reset()
            cur_profiling_path = f'{profiling_path_base}_config_{config_index}'
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            model = compile_model(URMA_P2P_BENCHMARK_GRAPH_Model(test_loop, 1, test_max_bs, ccl_buffer_size))
            loop_time = 2
            repeat_time = 1
            warm_time = 1
            with get_profiling(loop_time, repeat_time, warm_time, cur_profiling_path) as prof:
                for i_iter in range(loop_time):
                    y, receive_cnt = model(
                        x=x_npu,
                        group=hcomm_info,
                        world_size=test_world_size,
                        max_bs=test_max_bs,
                        mode=1,
                        ccl_buffer_size=ccl_buffer_size,
                        dst_rank_id_attr=dst_rank_id_val,
                        aiv_num=aiv_num,
                    )
                    if i_iter == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()

        dist.barrier()
        torch.npu.synchronize()
        torch.npu.empty_cache()

        results[config_index] = config_perf_info_list

        print(f'[INFO] device_{rank} Config [{config_index}] 完成, 收集到 {len(config_perf_info_list)} 条timing数据')

    dist.destroy_process_group()
    print(f'[INFO] rank {rank} 销毁通信域，多配置测试完成\n')

    queue.put([rank, results])

# endregion


# region Overhead+Latency测试

def run_overhead_latency_test(args):
    """运行Overhead+Latency测试 (mode=0)

    遍历args.data_sizes中的每个数据大小，测量各操作阶段时延。
    显示阶段分解: PROCESS_START->DUPLICATE_END, DUPLICATE_END->INIT_SQ_CQ_LOAD_END,
    INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END, SQE_TEMPLATE_END->PROCESS_TOKEN_END,
    PROCESS_TOKEN_END->P2P_SEND_END, P2P_SEND_END->DOORBELL_END,
    DOORBELL_END->POLL_CQ_END, POLL_CQ_END->UPDATE_PICI_END,
    PROCESS_START->UPDATE_PICI_END。
    包含抖动分析（基于total_us）。

    Returns:
        包含各数据大小测试结果的字典列表
    """
    print("\n" + "=" * 80)
    print("Overhead+Latency Test - mode=0")
    print("=" * 80)

    all_results = []

    # 预计算最大配置的HCCL_BUFFSIZE
    # OverheadLatency模式: 使用正常的bs/h分配，kernel只处理1个token
    # 不使用force_bs1，避免h过大导致UB溢出
    max_config_bs, max_config_h = data_size_to_bs_h(max(args.data_sizes), x_dtype)
    ccl_buffer_size_max = int(cal_hccl_buffsize(max_config_bs, max_config_h, world_size))
    os.environ['HCCL_BUFFSIZE'] = str(ccl_buffer_size_max)
    print(f"HCCL_BUFFSIZE: {os.environ['HCCL_BUFFSIZE']} (基于最大配置 bs={max_config_bs}, h={max_config_h})")

    dst_rank_id_val = args.dst_rank if args.dst_rank >= 0 else -1

    for data_size_bytes in args.data_sizes:
        bs, h = data_size_to_bs_h(data_size_bytes, x_dtype)
        max_bs = bs
        mode = 0  # overhead+latency mode

        print(f"\n--- 数据大小: {format_data_size(data_size_bytes)} ({data_size_bytes} bytes) ---")
        print(f"配置: bs={bs}, h={h}, max_bs={max_bs}, mode={mode}")
        print(f"迭代次数: {args.iterations}, 预热次数: {args.warmup}")

        # 当前配置的ccl_buffer_size（使用预计算的最大值）
        ccl_buffer_size = ccl_buffer_size_max

        cur_dst_rank_id_val = dst_rank_id_val
        x_per_rank = generate_p2p_inputs(bs, h, world_size)

        manager = Manager()
        result_queue = manager.Queue()

        processes = []
        for rank in range(world_size):
            cur_dst_rank = (rank + 1) % world_size if cur_dst_rank_id_val == -1 else cur_dst_rank_id_val
            p = Process(
                target=run_urma_p2p_benchmark_npu,
                args=(result_queue, rank, x_per_rank[rank], None,
                      max_bs, mode, world_size, cur_dst_rank, True, True,
                      args.iterations, args.warmup, None, ccl_buffer_size,
                      rank_per_dev, master_ip, graph_type, aiv_num, input_dtype)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 从各rank收集kernel打点数据（只收集sender rank的数据，receiver rank的P2P阶段时长为0）
        all_perf_info = []
        while not result_queue.empty():
            rank_result = result_queue.get()
            perf_info_list = rank_result[1][2]
            # Sender rank: InitRankComm有实际耗时，slot[2] > slot[1]（真正的P2P执行后cycle更大）
            # Receiver rank: todDstRankId==epRankId → slot[2] == slot[1](fallback) 或 slot[2]=0(旧kernel无fallback)
            # 条件 p[2] > p[1] 确保：sender通过(slot[2]>slot[1]), receiver排除(slot[2]<=slot[1])
            if perf_info_list and len(perf_info_list) > 0:
                # 打印每个rank的raw performance_info数据用于调试
                sample = perf_info_list[0]
                raw_rank = rank_result[0]
                print(f"[DEBUG] rank={raw_rank}, raw_perf_info_sample: "
                      f"[0]={sample[0]}, [1]={sample[1]}, [2]={sample[2]}, [3]={sample[3]}, "
                      f"[4]={sample[4]}, [5]={sample[5]}, [6]={sample[6]}, [7]={sample[7]}, [8]={sample[8]}")
                has_valid_p2p = any(p[2] > p[1] for p in perf_info_list)
                if has_valid_p2p:
                    all_perf_info.extend(perf_info_list)

        # 计算阶段分解（全部以us为单位）
        breakdown = compute_phase_breakdown(all_perf_info)

        result = {
            'test_type': 'latency',
            'data_size_bytes': data_size_bytes,
            'data_size_label': format_data_size(data_size_bytes),
            'iterations': args.iterations,
            'warmup': args.warmup,
            'bs': bs,
            'h': h,
            'max_bs': max_bs,
            'mode': mode,
            'world_size': world_size,
            'dst_rank': dst_rank_id_val if dst_rank_id_val >= 0 else 'auto',
            'dtype': x_dtype,
            'phase_breakdown': breakdown,
        }

        # 提取关键指标
        total_us_stats = breakdown.get('total_us', {})

        result['statistics'] = total_us_stats  # total_us统计作为主统计
        result['mean_total_us'] = total_us_stats.get('mean', 0)

        if total_us_stats:
            # 打印当前数据大小的详细结果 - 每个操作单独打印
            print(f"\n{'=' * 60}")
            print(f"Overhead+Latency Test Result - {format_data_size(data_size_bytes)}")
            print(f"{'=' * 60}")
            print(f"Data Size: {format_data_size(data_size_bytes)}")

            # 定义所有操作阶段的显示顺序和标签
            phase_display = [
                ('duplicate_us', 'PROCESS_START->DUPLICATE_END'),
                ('sq_cq_load_us', 'DUPLICATE_END->INIT_SQ_CQ_LOAD_END'),
                ('sqe_template_us', 'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END'),
                ('process_token_us', 'SQE_TEMPLATE_END->PROCESS_TOKEN_END'),
                ('p2p_send_us', 'PROCESS_TOKEN_END->P2P_SEND_END'),
                ('trigger_doorbell_us', 'P2P_SEND_END->DOORBELL_END'),
                ('poll_cq_us', 'DOORBELL_END->POLL_CQ_END'),
                ('update_pici_us', 'POLL_CQ_END->UPDATE_PICI_END'),
                ('total_us', 'PROCESS_START->UPDATE_PICI_END'),
            ]

            for phase_key, phase_label in phase_display:
                phase_stats = breakdown.get(phase_key, {})
                if phase_stats:
                    print(f"{phase_label} (us):")
                    print(f"  Mean: {phase_stats.get('mean', 0):.4f} us")
                    print(f"  P50: {phase_stats.get('p50', 0):.4f} us")
                    print(f"  P99: {phase_stats.get('p99', 0):.4f} us")

            print_phase_breakdown(breakdown, f"Overhead+Latency Phase Breakdown - {format_data_size(data_size_bytes)}")

            # 抖动分析（基于total_us）
            total_us_values = []
            for perf_info in all_perf_info:
                cycle_end = perf_info[8]
                cycle_start = perf_info[0]
                if cycle_end > cycle_start and cycle_start > 0:
                    total_us_values.append((cycle_end - cycle_start) / CYCLES_PER_US)

            if len(total_us_values) > 0:
                jitter_arr = np.array(total_us_values) - total_us_stats['mean']
                jitter_stats = {
                    'jitter_mean': np.mean(np.abs(jitter_arr)),
                    'jitter_max': np.max(np.abs(jitter_arr)),
                    'jitter_std': np.std(jitter_arr),
                }
                print(f"\nJitter Analysis (total_us):")
                print(f"  Mean jitter: {jitter_stats['jitter_mean']:.4f} us")
                print(f"  Max jitter: {jitter_stats['jitter_max']:.4f} us")
                print(f"  Jitter std: {jitter_stats['jitter_std']:.4f} us")

                result['jitter'] = jitter_stats

            print(f"{'=' * 60}")
        else:
            result['statistics'] = {}
            result['mean_total_us'] = 0
            result['raw_data'] = []
            print(f"[WARN] No kernel timing data found for Overhead+Latency test - {format_data_size(data_size_bytes)}")

        all_results.append(result)

    # 打印Overhead+Latency汇总表格
    print_overhead_latency_summary(all_results)

    return all_results


def print_overhead_latency_summary(results):
    """打印Overhead+Latency测试汇总表格"""
    print("\n" + "=" * 298)
    print("URMA P2P Overhead+Latency Test Summary - All Data Sizes")
    print("=" * 298)

    if not results:
        print("[WARN] 无有效测试结果")
        return

    header = f"{'No.':<5} {'Size':<10} {'PROCESS_START->DUPLICATE_END(us)':<38} {'DUPLICATE_END->INIT_SQ_CQ_LOAD_END(us)':<38} {'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END(us)':<38} {'SQE_TEMPLATE_END->PROCESS_TOKEN_END(us)':<38} {'PROCESS_TOKEN_END->P2P_SEND_END(us)':<38} {'P2P_SEND_END->DOORBELL_END(us)':<38} {'DOORBELL_END->POLL_CQ_END(us)':<38} {'POLL_CQ_END->UPDATE_PICI_END(us)':<38} {'PROCESS_START->UPDATE_PICI_END(us)':<38} {'JitterMean(us)':<16} {'JitterMax(us)':<16}"
    print(header)
    print("-" * 298)

    for i, r in enumerate(results):
        bd = r.get('phase_breakdown', {})
        duplicate_mean = bd.get('duplicate_us', {}).get('mean', 0)
        sq_cq_load_mean = bd.get('sq_cq_load_us', {}).get('mean', 0)
        sqe_template_mean = bd.get('sqe_template_us', {}).get('mean', 0)
        process_token_mean = bd.get('process_token_us', {}).get('mean', 0)
        p2p_send_mean = bd.get('p2p_send_us', {}).get('mean', 0)
        trigger_doorbell_mean = bd.get('trigger_doorbell_us', {}).get('mean', 0)
        poll_cq_mean = bd.get('poll_cq_us', {}).get('mean', 0)
        update_pici_mean = bd.get('update_pici_us', {}).get('mean', 0)
        total_us_mean = bd.get('total_us', {}).get('mean', 0)
        jitter_mean = r.get('jitter', {}).get('jitter_mean', 0)
        jitter_max = r.get('jitter', {}).get('jitter_max', 0)
        row = f"{i+1:<5} {r['data_size_label']:<10} {duplicate_mean:<38.4f} {sq_cq_load_mean:<38.4f} {sqe_template_mean:<38.4f} " \
              f"{process_token_mean:<38.4f} {p2p_send_mean:<38.4f} {trigger_doorbell_mean:<38.4f} " \
              f"{poll_cq_mean:<38.4f} {update_pici_mean:<38.4f} {total_us_mean:<38.4f} {jitter_mean:<16.4f} {jitter_max:<16.4f}"
        print(row)

    print("=" * 298)

# endregion


# region Bandwidth测试

def run_bandwidth_test(args):
    """运行带宽测试 (mode=1)

    遍历data_sizes列表中的每个大小，使用多配置共享通信域模式。
    使用kernel timing收集性能数据，替代msprof。
    P2P带宽计算: bandwidth_gbps = data_size_bytes / (mean_time_us / 1e6) / 1e9
    注意: P2P是单链路，不需要除以(world_size-1)

    Returns:
        包含各数据大小测试结果的字典列表
    """
    print("\n" + "=" * 80)
    print("Bandwidth Test - mode=1")
    print("=" * 80)
    print(f"数据大小数量: {len(args.data_sizes)}")
    print("-" * 80)

    all_results = []

    # 构建所有测试配置
    test_configs = []
    for data_size_bytes in args.data_sizes:
        bs, h = data_size_to_bs_h(data_size_bytes, x_dtype)
        max_bs = bs
        test_configs.append({
            'data_size_bytes': data_size_bytes,
            'bs': bs,
            'h': h,
            'max_bs': max_bs,
        })

    # 设置HCCL_BUFFSIZE为最大配置所需的大小
    max_config = max(test_configs, key=lambda c: c['max_bs'] * c['h'])
    os.environ['HCCL_BUFFSIZE'] = cal_hccl_buffsize(max_config['max_bs'], max_config['h'], world_size)
    print(f"HCCL_BUFFSIZE: {os.environ['HCCL_BUFFSIZE']} (基于最大配置 bs={max_config['max_bs']}, h={max_config['h']})")

    dst_rank_id_val = args.dst_rank if args.dst_rank >= 0 else -1

    # 生成所有配置的输入数据
    all_inputs = []
    for i, cfg in enumerate(test_configs):
        test_bs = cfg['bs']
        test_h = cfg['h']
        test_max_bs = cfg['max_bs']
        x_per_rank = generate_p2p_inputs(test_bs, test_h, world_size)
        cfg_ccl_buffer_size = int(cal_hccl_buffsize(test_max_bs, test_h, world_size))
        all_inputs.append((x_per_rank, test_max_bs, cfg['data_size_bytes'], i + 1, cfg_ccl_buffer_size))

    # 多配置共享通信域模式
    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        rank_configs = []
        for i, (x_per_rank, test_max_bs, test_data_size_bytes, config_index, cfg_ccl_buffer_size) in enumerate(all_inputs):
            cur_dst_rank = (rank + 1) % world_size if dst_rank_id_val == -1 else dst_rank_id_val
            rank_configs.append((
                x_per_rank[rank], test_max_bs, test_data_size_bytes, world_size,
                config_index, cur_dst_rank, args.iterations, args.warmup, cfg_ccl_buffer_size
            ))
        p = Process(
            target=run_urma_p2p_benchmark_npu_multi_config,
            args=(result_queue, rank, rank_configs,
                  rank_per_dev, master_ip, graph_type, aiv_num, input_dtype)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 收集各rank的结果：config_index -> performance_info_list（只收集sender rank的数据）
    config_perf_info = {}  # config_index -> list of performance_info arrays (from sender ranks only)
    while not result_queue.empty():
        rank, results = result_queue.get()
        if results is not None and isinstance(results, dict):
            for config_index, perf_info_list in results.items():
                # Sender rank: InitRankComm有实际耗时，slot[2] > slot[1]（真正的P2P执行后cycle更大）
                # Receiver rank: todDstRankId==epRankId → slot[2] == slot[1](fallback) 或 slot[2]=0(旧kernel无fallback)
                # 条件 p[2] > p[1] 确保：sender通过(slot[2]>slot[1]), receiver排除(slot[2]<=slot[1])
                if perf_info_list and len(perf_info_list) > 0:
                    # 打印每个rank/config的raw performance_info数据用于调试
                    sample = perf_info_list[0]
                    print(f"[DEBUG] rank={rank}, config={config_index}, raw_perf_info_sample: "
                          f"[0]={sample[0]}, [1]={sample[1]}, [2]={sample[2]}, [3]={sample[3]}, "
                          f"[4]={sample[4]}, [5]={sample[5]}, [6]={sample[6]}, [7]={sample[7]}, [8]={sample[8]}")
                    has_valid_p2p = any(p[2] > p[1] for p in perf_info_list)
                    if has_valid_p2p:
                        if config_index not in config_perf_info:
                            config_perf_info[config_index] = []
                        config_perf_info[config_index].extend(perf_info_list)
        else:
            print(f"[ERROR] Rank {rank} 多配置测试执行失败或返回无效数据")

    # 从kernel打点数据计算各配置的带宽和阶段分解
    for i, cfg in enumerate(test_configs):
        data_size_bytes = cfg['data_size_bytes']
        test_bs = cfg['bs']
        test_h = cfg['h']
        config_index = i + 1

        result = {
            'test_type': 'bandwidth',
            'data_size_bytes': data_size_bytes,
            'data_size_label': format_data_size(data_size_bytes),
            'iterations': args.iterations,
            'warmup': args.warmup,
            'bs': test_bs,
            'h': test_h,
            'max_bs': cfg['max_bs'],
            'mode': 1,
            'world_size': world_size,
            'dst_rank': dst_rank_id_val if dst_rank_id_val >= 0 else 'auto',
            'dtype': x_dtype,
        }

        if config_index in config_perf_info and len(config_perf_info[config_index]) > 0:
            # 计算阶段分解
            breakdown = compute_phase_breakdown(config_perf_info[config_index])
            result['phase_breakdown'] = breakdown

            # Bandwidth uses total execution time: total_us = [8]-[0] / 50
            total_us_stats = breakdown.get('total_us', {})

            if total_us_stats:
                result['statistics'] = total_us_stats

                mean_time_us = total_us_stats['mean']
                min_time_us = total_us_stats['min']

                # P2P带宽计算: 单链路，不除以(world_size-1)
                mean_time_s = mean_time_us / 1e6
                min_time_s = min_time_us / 1e6

                mean_bandwidth_gbps = (data_size_bytes / mean_time_s) / 1e9
                max_bandwidth_gbps = (data_size_bytes / min_time_s) / 1e9

                result['mean_bandwidth_gbps'] = mean_bandwidth_gbps
                result['max_bandwidth_gbps'] = max_bandwidth_gbps
                result['mean_time_us'] = mean_time_us
                result['min_time_us'] = min_time_us
                result['max_time_us'] = total_us_stats['max']
                result['mean_total_us'] = total_us_stats.get('mean', 0)

                # 打印当前配置的带宽结果和阶段分解
                print(f"\n  [{format_data_size(data_size_bytes)}] "
                      f"Mean BW: {mean_bandwidth_gbps:.2f} GB/s, "
                      f"Max BW: {max_bandwidth_gbps:.2f} GB/s, "
                      f"Mean Time: {mean_time_us:.2f} us")
                print_phase_breakdown(breakdown, f"Bandwidth Phase Breakdown - {format_data_size(data_size_bytes)}")
            else:
                result['statistics'] = {}
                result['mean_bandwidth_gbps'] = 0
                result['max_bandwidth_gbps'] = 0
                result['mean_time_us'] = 0
                result['min_time_us'] = 0
                result['max_time_us'] = 0
                result['mean_total_us'] = 0
                print(f"[WARN] Config [{config_index}] ({format_data_size(data_size_bytes)}) 无有效timing数据")
        else:
            result['phase_breakdown'] = {}
            result['statistics'] = {}
            result['mean_bandwidth_gbps'] = 0
            result['max_bandwidth_gbps'] = 0
            result['mean_time_us'] = 0
            result['min_time_us'] = 0
            result['max_time_us'] = 0
            result['mean_total_us'] = 0
            print(f"[WARN] Config [{config_index}] ({format_data_size(data_size_bytes)}) 未找到性能数据")

        all_results.append(result)

    # 打印带宽测试汇总
    print_bandwidth_summary(all_results)

    return all_results


def print_bandwidth_summary(results):
    """打印带宽测试汇总表格"""
    print("\n" + "=" * 318)
    print("URMA P2P Bandwidth Test Summary - All Configurations")
    print("=" * 318)

    if not results:
        print("[WARN] No valid test results")
        return

    header = f"{'No.':<5} {'Size':<10} {'Mean BW(GB/s)':<16} {'Max BW(GB/s)':<16} {'Mean Time(us)':<14} {'Min Time(us)':<14} {'PROCESS_START->DUPLICATE_END(us)':<38} {'DUPLICATE_END->INIT_SQ_CQ_LOAD_END(us)':<38} {'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END(us)':<38} {'SQE_TEMPLATE_END->PROCESS_TOKEN_END(us)':<38} {'PROCESS_TOKEN_END->P2P_SEND_END(us)':<38} {'P2P_SEND_END->DOORBELL_END(us)':<38} {'DOORBELL_END->POLL_CQ_END(us)':<38} {'POLL_CQ_END->UPDATE_PICI_END(us)':<38} {'PROCESS_START->UPDATE_PICI_END(us)':<38}"
    print(header)
    print("-" * 318)

    max_bandwidth = 0
    max_bw_result = None

    for i, r in enumerate(results):
        bd = r.get('phase_breakdown', {})
        duplicate_mean = bd.get('duplicate_us', {}).get('mean', 0)
        sq_cq_load_mean = bd.get('sq_cq_load_us', {}).get('mean', 0)
        sqe_template_mean = bd.get('sqe_template_us', {}).get('mean', 0)
        process_token_mean = bd.get('process_token_us', {}).get('mean', 0)
        p2p_send_mean = bd.get('p2p_send_us', {}).get('mean', 0)
        trigger_doorbell_mean = bd.get('trigger_doorbell_us', {}).get('mean', 0)
        poll_cq_mean = bd.get('poll_cq_us', {}).get('mean', 0)
        update_pici_mean = bd.get('update_pici_us', {}).get('mean', 0)
        total_us_mean = bd.get('total_us', {}).get('mean', 0)

        row = f"{i+1:<5} {r['data_size_label']:<10} " \
              f"{r.get('mean_bandwidth_gbps', 0):<16.2f} {r.get('max_bandwidth_gbps', 0):<16.2f} " \
              f"{r.get('mean_time_us', 0):<14.4f} {r.get('min_time_us', 0):<14.4f} " \
              f"{duplicate_mean:<38.4f} {sq_cq_load_mean:<38.4f} {sqe_template_mean:<38.4f} {process_token_mean:<38.4f} " \
              f"{p2p_send_mean:<38.4f} {trigger_doorbell_mean:<38.4f} {poll_cq_mean:<38.4f} {update_pici_mean:<38.4f} " \
              f"{total_us_mean:<38.4f}"
        print(row)

        if r.get('max_bandwidth_gbps', 0) > max_bandwidth:
            max_bandwidth = r['max_bandwidth_gbps']
            max_bw_result = r

    print("-" * 318)

    if max_bw_result:
        print(f"\n{'*' * 120}")
        print(f"Max Bandwidth: {max_bw_result['max_bandwidth_gbps']:.2f} GB/s")
        print(f"  Data Size: {max_bw_result['data_size_label']} ({max_bw_result['data_size_bytes']} bytes)")
        print(f"  Config: bs={max_bw_result['bs']}, h={max_bw_result['h']}")
        print(f"  Mean Time: {max_bw_result['mean_time_us']:.2f} us")
        bd = max_bw_result.get('phase_breakdown', {})
        total_us_mean = bd.get('total_us', {}).get('mean', 0)
        print(f"  Total: {total_us_mean:.4f} us")
        print(f"{'*' * 120}")

    # Bandwidth trend analysis
    print("\n" + "=" * 80)
    print("Bandwidth Trend Analysis")
    print("=" * 80)

    print(f"\n{'Data Size':<12} {'Mean BW(GB/s)':<18} {'Max BW(GB/s)':<18} {'Trend'}")
    print("-" * 80)

    prev_bw = None
    for r in results:
        trend = ""
        if prev_bw is not None:
            diff = r.get('max_bandwidth_gbps', 0) - prev_bw
            if diff > 0:
                trend = f"↑ +{diff:.2f}"
            elif diff < 0:
                trend = f"↓ {diff:.2f}"
            else:
                trend = "→ 0.00"
        prev_bw = r.get('max_bandwidth_gbps', 0)
        print(f"{r['data_size_label']:<12} {r.get('mean_bandwidth_gbps', 0):<18.2f} {r.get('max_bandwidth_gbps', 0):<18.2f} {trend}")

    print("=" * 80)

# endregion


# region CSV导出

def export_csv(results, output_csv):
    """导出测试结果为CSV

    CSV column names use snake_case for practical reasons (no special characters).
    Mapping of phase columns to their interval labels:
      duplicate_us_mean      -> PROCESS_START->DUPLICATE_END(us)
      sq_cq_load_us_mean     -> DUPLICATE_END->INIT_SQ_CQ_LOAD_END(us)
      sqe_template_us_mean   -> INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END(us)
      process_token_us_mean  -> SQE_TEMPLATE_END->PROCESS_TOKEN_END(us)
      p2p_send_us_mean       -> PROCESS_TOKEN_END->P2P_SEND_END(us)
      trigger_doorbell_us_mean -> P2P_SEND_END->DOORBELL_END(us)
      poll_cq_us_mean        -> DOORBELL_END->POLL_CQ_END(us)
      update_pici_us_mean    -> POLL_CQ_END->UPDATE_PICI_END(us)
      total_us_mean          -> PROCESS_START->UPDATE_PICI_END(us)
    """
    rows = []
    for r in results:
        stats = r.get('statistics', {})
        bd = r.get('phase_breakdown', {})
        row = {
            'test_type': r['test_type'],
            'data_size_bytes': r.get('data_size_bytes', 0),
            'data_size_label': format_data_size(r.get('data_size_bytes', 0)),
            'iterations': r.get('iterations', 0),
            'mean_time_us': stats.get('mean', 0) if stats else 0,
            'min_time_us': stats.get('min', 0) if stats else 0,
            'max_time_us': stats.get('max', 0) if stats else 0,
            'std_time_us': stats.get('std', 0) if stats else 0,
            'mean_bandwidth_gbps': r.get('mean_bandwidth_gbps', 0),
            'max_bandwidth_gbps': r.get('max_bandwidth_gbps', 0),
            'p50_time_us': stats.get('p50', 0) if stats else 0,
            'p90_time_us': stats.get('p90', 0) if stats else 0,
            'p99_time_us': stats.get('p99', 0) if stats else 0,
            'duplicate_us_mean': bd.get('duplicate_us', {}).get('mean', 0),
            'sq_cq_load_us_mean': bd.get('sq_cq_load_us', {}).get('mean', 0),
            'sqe_template_us_mean': bd.get('sqe_template_us', {}).get('mean', 0),
            'process_token_us_mean': bd.get('process_token_us', {}).get('mean', 0),
            'p2p_send_us_mean': bd.get('p2p_send_us', {}).get('mean', 0),
            'trigger_doorbell_us_mean': bd.get('trigger_doorbell_us', {}).get('mean', 0),
            'poll_cq_us_mean': bd.get('poll_cq_us', {}).get('mean', 0),
            'update_pici_us_mean': bd.get('update_pici_us', {}).get('mean', 0),
            'total_us_mean': bd.get('total_us', {}).get('mean', 0),
            'mean_total_us': r.get('mean_total_us', 0),
            'world_size': r.get('world_size', 0),
            'dst_rank': r.get('dst_rank', -1),
            'dtype': r.get('dtype', 'fp16'),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"结果已导出到: {output_csv}")

# endregion


# region 报告生成

def generate_report(results, output_dir, args):
    """生成文本测试报告

    Args:
        results: 测试结果列表
        output_dir: 输出目录
        args: 命行参数
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'urma_p2p_benchmark_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("URMA P2P Benchmark 测试报告\n")
        f.write("=" * 80 + "\n\n")

        # 环境信息
        f.write("1. 环境信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"测试类型: {args.test_type}\n")
        f.write(f"数据类型: {args.dtype}\n")
        f.write(f"world_size: {args.world_size}\n")
        f.write(f"rank_per_dev: {args.rank_per_dev}\n")
        f.write(f"服务器数量: {args.server_num}\n")
        f.write(f"图模式: {args.graph_type}\n")
        f.write(f"目标rank: {args.dst_rank}\n")
        f.write(f"迭代次数: {args.iterations}\n")
        f.write(f"预热次数: {args.warmup}\n")
        f.write(f"AIV核数: {aiv_num}\n")
        f.write(f"通信模式: SQE级HcclCombinOpParam URMA\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"时延单位: us (1 us = 50 cycles on Ascend AIV)\n")
        f.write("\n")

        # 各测试结果
        test_type_labels = {
            'latency': 'Overhead+Latency',
            'bandwidth': 'Bandwidth',
        }
        for r in results:
            label = test_type_labels.get(r['test_type'], r['test_type'])
            f.write(f"\n2. {label} 测试结果\n")
            f.write("-" * 40 + "\n")
            f.write(f"数据大小: {r['data_size_label']} ({r['data_size_bytes']} bytes)\n")
            f.write(f"配置: bs={r.get('bs', 'N/A')}, h={r.get('h', 'N/A')}, max_bs={r.get('max_bs', 'N/A')}\n")
            f.write(f"迭代次数: {r.get('iterations', 'N/A')}\n")

            bd = r.get('phase_breakdown', {})
            if bd:
                phase_labels = {
                    'duplicate_us':           'PROCESS_START->DUPLICATE_END',
                    'sq_cq_load_us':          'DUPLICATE_END->INIT_SQ_CQ_LOAD_END',
                    'sqe_template_us':        'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END',
                    'process_token_us':       'SQE_TEMPLATE_END->PROCESS_TOKEN_END',
                    'p2p_send_us':            'PROCESS_TOKEN_END->P2P_SEND_END',
                    'trigger_doorbell_us':    'P2P_SEND_END->DOORBELL_END',
                    'poll_cq_us':             'DOORBELL_END->POLL_CQ_END',
                    'update_pici_us':         'POLL_CQ_END->UPDATE_PICI_END',
                    'total_us':               'PROCESS_START->UPDATE_PICI_END',
                }
                for phase_name, label in phase_labels.items():
                    stats = bd.get(phase_name, {})
                    if stats:
                        f.write(f"  {label}: mean={stats.get('mean', 0):.4f} us, "
                                f"min={stats.get('min', 0):.4f} us, "
                                f"max={stats.get('max', 0):.4f} us, "
                                f"std={stats.get('std', 0):.4f} us, "
                                f"P50={stats.get('p50', 0):.4f} us, "
                                f"P90={stats.get('p90', 0):.4f} us, "
                                f"P99={stats.get('p99', 0):.4f} us\n")

            if r.get('mean_bandwidth_gbps', 0) > 0:
                f.write(f"Mean Bandwidth: {r['mean_bandwidth_gbps']:.2f} GB/s\n")
                f.write(f"Max Bandwidth: {r['max_bandwidth_gbps']:.2f} GB/s\n")

            if r.get('jitter'):
                jitter = r['jitter']
                f.write(f"\nJitter Analysis:\n")
                f.write(f"  Mean jitter: {jitter['jitter_mean']:.4f} us\n")
                f.write(f"  Max jitter: {jitter['jitter_max']:.4f} us\n")
                f.write(f"  Jitter std: {jitter['jitter_std']:.4f} us\n")

            f.write("\n")

        # Summary table
        f.write("\n3. Summary Table\n")
        f.write("-" * 318 + "\n")
        header = f"{'Type':<16} {'Size':<10} {'PROCESS_START->DUPLICATE_END(us)':<38} {'DUPLICATE_END->INIT_SQ_CQ_LOAD_END(us)':<38} {'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END(us)':<38} {'SQE_TEMPLATE_END->PROCESS_TOKEN_END(us)':<38} {'PROCESS_TOKEN_END->P2P_SEND_END(us)':<38} {'P2P_SEND_END->DOORBELL_END(us)':<38} {'DOORBELL_END->POLL_CQ_END(us)':<38} {'POLL_CQ_END->UPDATE_PICI_END(us)':<38} {'PROCESS_START->UPDATE_PICI_END(us)':<38} {'Mean BW(GB/s)':<16} {'Max BW(GB/s)':<16}\n"
        f.write(header)
        f.write("-" * 318 + "\n")
        test_type_labels = {
            'latency': 'Overhead+Latency',
            'bandwidth': 'Bandwidth',
        }
        for r in results:
            bd = r.get('phase_breakdown', {})
            label = test_type_labels.get(r['test_type'], r['test_type'])
            row = f"{label:<16} {r['data_size_label']:<10} " \
                  f"{bd.get('duplicate_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('sq_cq_load_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('sqe_template_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('process_token_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('p2p_send_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('trigger_doorbell_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('poll_cq_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('update_pici_us', {}).get('mean', 0):<38.4f} " \
                  f"{bd.get('total_us', {}).get('mean', 0):<38.4f} " \
                  f"{r.get('mean_bandwidth_gbps', 0):<16.2f} " \
                  f"{r.get('max_bandwidth_gbps', 0):<16.2f}\n"
            f.write(row)
        f.write("-" * 318 + "\n")

    print(f"测试报告已保存到: {report_path}")

# endregion


# region 结果汇总打印

def print_summary(results):
    """Print final test results summary"""
    print("\n" + "=" * 80)
    print("URMA P2P Benchmark Summary")
    print("=" * 80)

    # Define all operation phases for display
    phase_display = [
        ('duplicate_us', 'PROCESS_START->DUPLICATE_END'),
        ('sq_cq_load_us', 'DUPLICATE_END->INIT_SQ_CQ_LOAD_END'),
        ('sqe_template_us', 'INIT_SQ_CQ_LOAD_END->SQE_TEMPLATE_END'),
        ('process_token_us', 'SQE_TEMPLATE_END->PROCESS_TOKEN_END'),
        ('p2p_send_us', 'PROCESS_TOKEN_END->P2P_SEND_END'),
        ('trigger_doorbell_us', 'P2P_SEND_END->DOORBELL_END'),
        ('poll_cq_us', 'DOORBELL_END->POLL_CQ_END'),
        ('update_pici_us', 'POLL_CQ_END->UPDATE_PICI_END'),
        ('total_us', 'PROCESS_START->UPDATE_PICI_END'),
    ]

    for r in results:
        test_type = r['test_type']
        bd = r.get('phase_breakdown', {})

        if test_type == 'latency':
            print(f"\n[Overhead+Latency Test - {r['data_size_label']}]")
            total_us = bd.get('total_us', {})
            if total_us:
                print(f"  Total: mean={total_us.get('mean', 0):.4f} us, "
                      f"P50={total_us.get('p50', 0):.4f} us, "
                      f"P90={total_us.get('p90', 0):.4f} us, "
                      f"P99={total_us.get('p99', 0):.4f} us")
            for phase_key, phase_label in phase_display:
                if phase_key == 'total_us':
                    continue  # already printed above
                phase_mean = bd.get(phase_key, {}).get('mean', 0)
                if phase_mean > 0:
                    print(f"  {phase_label}: {phase_mean:.4f} us")
            if r.get('jitter'):
                print(f"  Mean jitter: {r['jitter']['jitter_mean']:.4f} us")

        elif test_type == 'bandwidth':
            bw = r.get('mean_bandwidth_gbps', 0)
            max_bw = r.get('max_bandwidth_gbps', 0)
            if bw > 0:
                print(f"\n[Bandwidth Test - {r['data_size_label']}]")
                print(f"  Mean BW: {bw:.2f} GB/s")
                print(f"  Max BW: {max_bw:.2f} GB/s")
                print(f"  Mean Time: {r.get('mean_time_us', 0):.2f} us")
                total_us = bd.get('total_us', {}).get('mean', 0)
                print(f"  Total: {total_us:.4f} us")

    # Find max bandwidth
    bandwidth_results = [r for r in results if r['test_type'] == 'bandwidth' and r.get('max_bandwidth_gbps', 0) > 0]
    if bandwidth_results:
        best = max(bandwidth_results, key=lambda x: x['max_bandwidth_gbps'])
        print(f"\n{'*' * 60}")
        print(f"Peak P2P Bandwidth: {best['max_bandwidth_gbps']:.2f} GB/s (Data Size: {best['data_size_label']})")
        print(f"{'*' * 60}")

    # Find minimum total time
    latency_results = [r for r in results if r['test_type'] == 'latency']
    if latency_results:
        smallest = min(latency_results, key=lambda x: x.get('mean_total_us', float('inf')))
        print(f"\nMinimum Total: {smallest.get('mean_total_us', 0):.4f} us "
              f"(Data Size: {smallest['data_size_label']})")

    print("\n" + "=" * 80)

# endregion


# region 主函数

def main():
    args = parse_args()
    setup_environment(args)

    results = []

    if args.test_type in ['latency', 'all']:
        results.extend(run_overhead_latency_test(args))
    if args.test_type in ['bandwidth', 'all']:
        results.extend(run_bandwidth_test(args))

    export_csv(results, args.output_csv)
    generate_report(results, args.output_dir, args)
    print_summary(results)

    print(f"\n测试完成! 共执行 {len(results)} 个测试配置")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()