"""
URMA P2P基准测试工具（SQE级HcclCombinOpParam URMA通信）

支持三种测试模式:
- overhead (mode=0): 头开销测试，测量URMA通信的固定启动成本
- latency (mode=1): 时延测试，测量P2P小包通信时延及百分位分布
- bandwidth (mode=2): 带宽测试，测量P2P单链路吞吐量

注意: 本工具使用SQE级HcclCombinOpParam URMA通信，算子通过
.MC2().HcclGroup({"group"})声明HCCL通信域，框架自动创建HCCL上下文，
无需手动传递mc2Context。算子通过npu_ops_transformer.ops.npu_urma_p2p_benchmark调用。
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
    parser.add_argument('--test_type', choices=['overhead', 'latency', 'bandwidth', 'all'], default='all',
                        help='测试类型: overhead/latency/bandwidth/all')
    parser.add_argument('--data_sizes', type=int, nargs='+',
                        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152],
                        help='数据大小列表(字节), 从256B到2MB')
    parser.add_argument('--latency_packet_size', type=int, default=64, help='时延测试数据包大小(字节)')
    parser.add_argument('--iterations', type=int, default=100, help='带宽/开销迭代次数')
    parser.add_argument('--latency_iterations', type=int, default=1000, help='时延迭代次数')
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
    print(f"时延迭代次数: {args.latency_iterations}")
    print(f"目标rank: {args.dst_rank}")
    print(f"数据大小列表: {args.data_sizes}")
    print(f"时延包大小: {args.latency_packet_size}")
    print(f"输出CSV: {args.output_csv}")
    print(f"输出目录: {args.output_dir}")
    print(f"通信模式: SQE级HcclCombinOpParam URMA")
    print(f"{'=' * 60}\n")

# endregion


# region 数据大小映射函数

def data_size_to_bs_h(data_size_bytes, dtype='fp16'):
    """将数据字节大小转换为(bs, h)参数对

    Args:
        data_size_bytes: 数据大小(字节)
        dtype: 数据类型('fp16'或'bf16')

    Returns:
        (bs, h) 参数对, bs为batch size, h为hidden size元素数
    """
    dtype_size = 2  # fp16和bf16都是2字节
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

    performance_info为int64 tensor [6]，6个timing slot:
      [0] PROCESS_START    - Process()开始（dstRankIds填充前）
      [1] INIT_RANK_COMM_END - InitRankComm完成（SQE组装完成）
      [2] P2P_SEND_END     - P2pSend+P2pSendDrain完成（数据发送完成）
      [3] SETSTATUS_END    - SetStatus完成
      [4] WAITDISPATCH_END - WaitDispatch完成
      [5] TOTAL_END        - Process()完全结束

    各阶段时延计算(cycle差 * 20 = ns, cycle差 / 50 = us):
      sqe_assembly_ns:         [1]-[0] * 20  (SQE组装开销, ns)
      p2p_send_ns:             [2]-[1] * 20  (纯P2P数据传输, ns)
      communication_latency_ns: [2]-[0] * 20 (通信总时延 = overhead + P2P, ns)
      setstatus_ns:            [3]-[2] * 20  (SetStatus, ns)
      waitdispatch_ns:         [4]-[3] * 20  (WaitDispatch, ns)
      local_copy_ns:           [5]-[4] * 20  (LocalWindowCopy, ns)
      total_ns:                [5]-[0] * 20  (总耗时, ns)
      total_us:                [5]-[0] / 50  (总耗时, us)

    Args:
        perf_info_list: 每次迭代的performance_info numpy数组列表，每个为shape [6]

    Returns:
        dict，每个phase包含 mean/min/max/std/p50/p90/p99 统计值（ns单位）
        以及 total_us 阶段额外包含 us 单位统计
    """
    # 各阶段名称 -> (end_index, start_index) 映射
    phases = {
        'sqe_assembly_ns':         (1, 0),
        'p2p_send_ns':             (2, 1),
        'communication_latency_ns': (2, 0),
        'setstatus_ns':            (3, 2),
        'waitdispatch_ns':         (4, 3),
        'local_copy_ns':           (5, 4),
        'total_ns':                (5, 0),
    }

    result = {}

    for phase_name, (end_idx, start_idx) in phases.items():
        values_ns = []
        for perf_info in perf_info_list:
            cycle_end = perf_info[end_idx]
            cycle_start = perf_info[start_idx]
            if cycle_end > cycle_start:
                values_ns.append((cycle_end - cycle_start) * NS_PER_CYCLE)

        if len(values_ns) > 0:
            arr = np.array(values_ns)
            result[phase_name] = compute_statistics(arr)
        else:
            result[phase_name] = {}

    # total_us: 总耗时以us为单位 ([5]-[0] / 50)
    values_us = []
    for perf_info in perf_info_list:
        cycle_end = perf_info[5]
        cycle_start = perf_info[0]
        if cycle_end > cycle_start:
            values_us.append((cycle_end - cycle_start) / CYCLES_PER_US)

    if len(values_us) > 0:
        arr = np.array(values_us)
        result['total_us'] = compute_statistics(arr)
    else:
        result['total_us'] = {}

    return result


def print_phase_breakdown(breakdown, title="Phase Breakdown"):
    """打印阶段分解表格

    Args:
        breakdown: compute_phase_breakdown返回的dict
        title: 表格标题
    """
    phase_labels = {
        'sqe_assembly_ns':         'SQE组装',
        'p2p_send_ns':             'P2P数据传输',
        'communication_latency_ns': '通信总时延',
        'setstatus_ns':            'SetStatus',
        'waitdispatch_ns':         'WaitDispatch',
        'local_copy_ns':           'LocalWindowCopy',
        'total_ns':                '总耗时(ns)',
        'total_us':                '总耗时(us)',
    }

    print(f"\n  {title}")
    print(f"  {'阶段':<22} {'Mean':<12} {'Min':<12} {'Max':<12} {'Std':<12} {'P50':<12} {'P90':<12} {'P99':<12}")
    print(f"  {'-' * 106}")

    for phase_name, label in phase_labels.items():
        stats = breakdown.get(phase_name, {})
        if not stats:
            continue
        unit = 'us' if phase_name == 'total_us' else 'ns'
        print(f"  {label:<22} {stats.get('mean', 0):.2f}{unit:<6} "
              f"{stats.get('min', 0):.2f}{unit:<6} "
              f"{stats.get('max', 0):.2f}{unit:<6} "
              f"{stats.get('std', 0):.2f}{unit:<6} "
              f"{stats.get('p50', 0):.2f}{unit:<6} "
              f"{stats.get('p90', 0):.2f}{unit:<6} "
              f"{stats.get('p99', 0):.2f}{unit:<6}")

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
        mode: 测试模式 (0=overhead, 1=latency, 2=bandwidth)
        ccl_buffer_size: HCCL缓冲区大小
        dst_rank_id_attr: P2P目标rank属性 (-1=全部rank)
        aiv_num: AIV核数

    Returns:
        (y, receive_cnt, performance_info) 输出tensor三元组
        performance_info: int64 tensor [6], kernel内部各阶段cycle计数
        [0]=PROCESS_START, [1]=INIT_RANK_COMM_END, [2]=P2P_SEND_END,
        [3]=SETSTATUS_END, [4]=WAITDISPATCH_END, [5]=TOTAL_END
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
        test_mode: 测试模式(0=overhead, 1=latency, 2=bandwidth)
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
    # performance_info: int64 [6]
    # [0]=PROCESS_START, [1]=INIT_RANK_COMM_END, [2]=P2P_SEND_END,
    # [3]=SETSTATUS_END, [4]=WAITDISPATCH_END, [5]=TOTAL_END
    # CYCLES_PER_US = 50, NS_PER_CYCLE = 20

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
              f"max_bs={test_max_bs}, dst_rank={dst_rank_id_val}, mode=2(带宽)")
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
                    mode=2,  # bandwidth mode
                    ccl_buffer_size=ccl_buffer_size,
                    dst_rank_id_attr=dst_rank_id_val,
                    aiv_num=aiv_num,
                )
                torch.npu.synchronize()
                config_perf_info_list.append(perf_info.cpu().numpy())
        else:
            # 图模式暂不支持kernel打点，保留原有msprof方式
            torch._dynamo.reset()
            cur_profiling_path = f'{profiling_path_base}_config_{config_index}'
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            model = compile_model(URMA_P2P_BENCHMARK_GRAPH_Model(test_loop, 2, test_max_bs, ccl_buffer_size))
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
                        mode=2,
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


# region Overhead测试

def run_overhead_test(args):
    """运行头开销测试 (mode=0)

    遍历args.data_sizes中的每个数据大小，测量URMA通信的各阶段开销(ns)。
    显示阶段分解: SQE组装([1]-[0]), P2P发送([2]-[1]), 总耗时([5]-[0])，均以ns为单位。

    Returns:
        包含各数据大小测试结果的字典列表
    """
    print("\n" + "=" * 80)
    print("Overhead Test (头开销测试) - mode=0")
    print("=" * 80)

    all_results = []

    # 预计算最大配置的HCCL_BUFFSIZE
    max_bs_needed = max(data_size_to_bs_h(ds, x_dtype)[0] for ds in args.data_sizes)
    max_h_needed = max(data_size_to_bs_h(ds, x_dtype)[1] for ds in args.data_sizes)
    ccl_buffer_size_max = int(cal_hccl_buffsize(max_bs_needed, max_h_needed, world_size))
    os.environ['HCCL_BUFFSIZE'] = str(ccl_buffer_size_max)
    print(f"HCCL_BUFFSIZE: {os.environ['HCCL_BUFFSIZE']} (基于最大配置 bs={max_bs_needed}, h={max_h_needed})")

    dst_rank_id_val = args.dst_rank if args.dst_rank >= 0 else -1

    for data_size_bytes in args.data_sizes:
        bs, h = data_size_to_bs_h(data_size_bytes, x_dtype)
        max_bs = bs
        mode = 0  # overhead mode

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

        # 从各rank收集kernel打点数据
        all_perf_info = []
        while not result_queue.empty():
            rank_result = result_queue.get()
            all_perf_info.extend(rank_result[1][2])  # performance_info_list

        # 计算阶段分解（全部以ns为单位）
        breakdown = compute_phase_breakdown(all_perf_info)

        result = {
            'test_type': 'overhead',
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
        comm_lat_stats = breakdown.get('communication_latency_ns', {})
        total_ns_stats = breakdown.get('total_ns', {})
        total_us_stats = breakdown.get('total_us', {})

        result['statistics'] = comm_lat_stats  # 通信总时延统计作为主统计
        result['mean_comm_latency_ns'] = comm_lat_stats.get('mean', 0)
        result['min_comm_latency_ns'] = comm_lat_stats.get('min', 0)
        result['max_comm_latency_ns'] = comm_lat_stats.get('max', 0)
        result['mean_total_ns'] = total_ns_stats.get('mean', 0)
        result['mean_total_us'] = total_us_stats.get('mean', 0)

        # 打印当前数据大小的overhead结果
        print(f"\n{'=' * 60}")
        print(f"Overhead Test 结果 - {format_data_size(data_size_bytes)}")
        print(f"{'=' * 60}")
        if comm_lat_stats:
            print(f"通信总时延 (ns):")
            print(f"  平均: {comm_lat_stats['mean']:.2f} ns")
            print(f"  最小: {comm_lat_stats['min']:.2f} ns")
            print(f"  最大: {comm_lat_stats['max']:.2f} ns")
            print(f"  标准差: {comm_lat_stats['std']:.2f} ns")
            print(f"  P50: {comm_lat_stats['p50']:.2f} ns")
            print(f"  P90: {comm_lat_stats['p90']:.2f} ns")
            print(f"  P99: {comm_lat_stats['p99']:.2f} ns")
        if total_ns_stats:
            print(f"总耗时 (ns):")
            print(f"  平均: {total_ns_stats['mean']:.2f} ns ({total_us_stats.get('mean', 0):.2f} us)")

        print_phase_breakdown(breakdown, f"Overhead Phase Breakdown - {format_data_size(data_size_bytes)}")
        print(f"{'=' * 60}")

        all_results.append(result)

    # 打印overhead汇总表格
    print_overhead_summary(all_results)

    return all_results


def print_overhead_summary(results):
    """打印overhead测试汇总表格"""
    print("\n" + "=" * 100)
    print("URMA P2P Overhead Test Summary - All Data Sizes")
    print("=" * 100)

    if not results:
        print("[WARN] 无有效测试结果")
        return

    header = f"{'No.':<5} {'Size':<10} {'SQE组装(ns)':<16} {'P2P传输(ns)':<16} {'通信总时延(ns)':<18} {'总耗时(ns)':<16} {'总耗时(us)':<16}"
    print(header)
    print("-" * 100)

    for i, r in enumerate(results):
        bd = r.get('phase_breakdown', {})
        sqe_mean = bd.get('sqe_assembly_ns', {}).get('mean', 0)
        p2p_mean = bd.get('p2p_send_ns', {}).get('mean', 0)
        comm_mean = bd.get('communication_latency_ns', {}).get('mean', 0)
        total_ns_mean = bd.get('total_ns', {}).get('mean', 0)
        total_us_mean = bd.get('total_us', {}).get('mean', 0)
        row = f"{i+1:<5} {r['data_size_label']:<10} {sqe_mean:<16.2f} {p2p_mean:<16.2f} " \
              f"{comm_mean:<18.2f} {total_ns_mean:<16.2f} {total_us_mean:<16.2f}"
        print(row)

    print("=" * 100)

# endregion


# region Latency测试

def run_latency_test(args):
    """运行时延测试 (mode=1)

    使用latency_packet_size(64B)测量P2P小包通信时延及百分位分布。
    显示阶段分解: SQE组装([1]-[0]), P2P发送+drain([2]-[1]), 通信总时延([2]-[0])，均以ns为单位。

    Returns:
        包含测试结果的字典列表
    """
    print("\n" + "=" * 80)
    print("Latency Test (时延测试) - mode=1")
    print("=" * 80)

    data_size_bytes = args.latency_packet_size
    bs, h = data_size_to_bs_h(data_size_bytes, x_dtype)
    max_bs = bs
    mode = 1  # latency mode
    latency_warmup = 20

    print(f"数据大小: {format_data_size(data_size_bytes)} ({data_size_bytes} bytes)")
    print(f"配置: bs={bs}, h={h}, max_bs={max_bs}, mode={mode}")
    print(f"迭代次数: {args.latency_iterations}, 预热次数: {latency_warmup}")

    # 设置HCCL_BUFFSIZE
    ccl_buffer_size = int(cal_hccl_buffsize(max_bs, h, world_size))
    os.environ['HCCL_BUFFSIZE'] = str(ccl_buffer_size)
    print(f"HCCL_BUFFSIZE: {os.environ['HCCL_BUFFSIZE']}")

    dst_rank_id_val = args.dst_rank if args.dst_rank >= 0 else -1

    x_per_rank = generate_p2p_inputs(bs, h, world_size)

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        cur_dst_rank = (rank + 1) % world_size if dst_rank_id_val == -1 else dst_rank_id_val
        p = Process(
            target=run_urma_p2p_benchmark_npu,
            args=(result_queue, rank, x_per_rank[rank], None,
                  max_bs, mode, world_size, cur_dst_rank, True, True,
                  args.latency_iterations, latency_warmup, None, ccl_buffer_size,
                  rank_per_dev, master_ip, graph_type, aiv_num, input_dtype)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 从各rank收集kernel打点数据
    all_perf_info = []
    while not result_queue.empty():
        rank_result = result_queue.get()
        all_perf_info.extend(rank_result[1][2])  # performance_info_list

    # 计算阶段分解（全部以ns为单位）
    breakdown = compute_phase_breakdown(all_perf_info)

    result = {
        'test_type': 'latency',
        'data_size_bytes': data_size_bytes,
        'data_size_label': format_data_size(data_size_bytes),
        'iterations': args.latency_iterations,
        'warmup': latency_warmup,
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
    sqe_stats = breakdown.get('sqe_assembly_ns', {})
    p2p_stats = breakdown.get('p2p_send_ns', {})
    comm_lat_stats = breakdown.get('communication_latency_ns', {})
    total_ns_stats = breakdown.get('total_ns', {})
    total_us_stats = breakdown.get('total_us', {})

    result['statistics'] = comm_lat_stats  # 通信总时延统计作为主统计
    result['mean_comm_latency_ns'] = comm_lat_stats.get('mean', 0)
    result['min_comm_latency_ns'] = comm_lat_stats.get('min', 0)
    result['max_comm_latency_ns'] = comm_lat_stats.get('max', 0)
    result['mean_total_ns'] = total_ns_stats.get('mean', 0)
    result['mean_total_us'] = total_us_stats.get('mean', 0)

    if comm_lat_stats:
        print(f"\n{'=' * 60}")
        print("Latency Test 结果")
        print(f"{'=' * 60}")
        print(f"数据大小: {format_data_size(data_size_bytes)}")
        print(f"通信总时延 (ns):")
        print(f"  平均: {comm_lat_stats['mean']:.2f} ns")
        print(f"  最小: {comm_lat_stats['min']:.2f} ns")
        print(f"  最大: {comm_lat_stats['max']:.2f} ns")
        print(f"  标准差: {comm_lat_stats['std']:.2f} ns")
        print(f"  P50: {comm_lat_stats['p50']:.2f} ns")
        print(f"  P90: {comm_lat_stats['p90']:.2f} ns")
        print(f"  P99: {comm_lat_stats['p99']:.2f} ns")

        print_phase_breakdown(breakdown, f"Latency Phase Breakdown - {format_data_size(data_size_bytes)}")

        # 抖动分析（基于通信总时延ns）
        comm_latency_values = []
        for perf_info in all_perf_info:
            cycle_end = perf_info[2]
            cycle_start = perf_info[0]
            if cycle_end > cycle_start:
                comm_latency_values.append((cycle_end - cycle_start) * NS_PER_CYCLE)

        if len(comm_latency_values) > 0:
            jitter_arr = np.array(comm_latency_values) - comm_lat_stats['mean']
            jitter_stats = {
                'jitter_mean': np.mean(np.abs(jitter_arr)),
                'jitter_max': np.max(np.abs(jitter_arr)),
                'jitter_std': np.std(jitter_arr),
            }
            print(f"\n抖动分析 (通信总时延):")
            print(f"  平均抖动: {jitter_stats['jitter_mean']:.2f} ns")
            print(f"  最大抖动: {jitter_stats['jitter_max']:.2f} ns")
            print(f"  抖动标准差: {jitter_stats['jitter_std']:.2f} ns")

            result['jitter'] = jitter_stats

        print(f"{'=' * 60}")
    else:
        result['statistics'] = {}
        result['mean_comm_latency_ns'] = 0
        result['min_comm_latency_ns'] = 0
        result['max_comm_latency_ns'] = 0
        result['mean_total_ns'] = 0
        result['mean_total_us'] = 0
        result['raw_data'] = []
        print("[WARN] 未找到latency测试的kernel打点数据")

    return [result]

# endregion


# region Bandwidth测试

def run_bandwidth_test(args):
    """运行带宽测试 (mode=2)

    遍历data_sizes列表中的每个大小，使用多配置共享通信域模式。
    使用kernel timing收集性能数据，替代msprof。
    P2P带宽计算: bandwidth_gbps = data_size_bytes / (mean_time_us / 1e6) / 1e9
    注意: P2P是单链路，不需要除以(world_size-1)

    Returns:
        包含各数据大小测试结果的字典列表
    """
    print("\n" + "=" * 80)
    print("Bandwidth Test (带宽测试) - mode=2")
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

    # 收集各rank的结果：config_index -> performance_info_list
    config_perf_info = {}  # config_index -> list of performance_info arrays (来自所有rank)
    while not result_queue.empty():
        rank, results = result_queue.get()
        if results is not None and isinstance(results, dict):
            for config_index, perf_info_list in results.items():
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
            'mode': 2,
            'world_size': world_size,
            'dst_rank': dst_rank_id_val if dst_rank_id_val >= 0 else 'auto',
            'dtype': x_dtype,
        }

        if config_index in config_perf_info and len(config_perf_info[config_index]) > 0:
            # 计算阶段分解
            breakdown = compute_phase_breakdown(config_perf_info[config_index])
            result['phase_breakdown'] = breakdown

            # 带宽使用总执行时间(和bandwidth_test工具一致): total_us = [5]-[0] / 50
            total_us_stats = breakdown.get('total_us', {})
            total_ns_stats = breakdown.get('total_ns', {})

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
                result['mean_total_ns'] = total_ns_stats.get('mean', 0)

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
                result['mean_total_ns'] = 0
                print(f"[WARN] Config [{config_index}] ({format_data_size(data_size_bytes)}) 无有效timing数据")
        else:
            result['phase_breakdown'] = {}
            result['statistics'] = {}
            result['mean_bandwidth_gbps'] = 0
            result['max_bandwidth_gbps'] = 0
            result['mean_time_us'] = 0
            result['min_time_us'] = 0
            result['max_time_us'] = 0
            result['mean_total_ns'] = 0
            print(f"[WARN] Config [{config_index}] ({format_data_size(data_size_bytes)}) 未找到性能数据")

        all_results.append(result)

    # 打印带宽测试汇总
    print_bandwidth_summary(all_results)

    return all_results


def print_bandwidth_summary(results):
    """打印带宽测试汇总表格"""
    print("\n" + "=" * 120)
    print("URMA P2P Bandwidth Test Summary - All Configurations")
    print("=" * 120)

    if not results:
        print("[WARN] 无有效测试结果")
        return

    header = f"{'No.':<5} {'Size':<10} {'Mean BW(GB/s)':<16} {'Max BW(GB/s)':<16} {'Mean Time(us)':<14} {'Min Time(us)':<14} {'SQE(ns)':<12} {'P2P(ns)':<12} {'CommLat(ns)':<14} {'Total(ns)':<12}"
    print(header)
    print("-" * 120)

    max_bandwidth = 0
    max_bw_result = None

    for i, r in enumerate(results):
        bd = r.get('phase_breakdown', {})
        sqe_mean = bd.get('sqe_assembly_ns', {}).get('mean', 0)
        p2p_mean = bd.get('p2p_send_ns', {}).get('mean', 0)
        comm_mean = bd.get('communication_latency_ns', {}).get('mean', 0)
        total_ns_mean = bd.get('total_ns', {}).get('mean', 0)

        row = f"{i+1:<5} {r['data_size_label']:<10} " \
              f"{r.get('mean_bandwidth_gbps', 0):<16.2f} {r.get('max_bandwidth_gbps', 0):<16.2f} " \
              f"{r.get('mean_time_us', 0):<14.2f} {r.get('min_time_us', 0):<14.2f} " \
              f"{sqe_mean:<12.2f} {p2p_mean:<12.2f} {comm_mean:<14.2f} {total_ns_mean:<12.2f}"
        print(row)

        if r.get('max_bandwidth_gbps', 0) > max_bandwidth:
            max_bandwidth = r['max_bandwidth_gbps']
            max_bw_result = r

    print("-" * 120)

    if max_bw_result:
        print(f"\n{'*' * 120}")
        print(f"最大带宽: {max_bw_result['max_bandwidth_gbps']:.2f} GB/s")
        print(f"  数据大小: {max_bw_result['data_size_label']} ({max_bw_result['data_size_bytes']} bytes)")
        print(f"  配置: bs={max_bw_result['bs']}, h={max_bw_result['h']}")
        print(f"  平均时间: {max_bw_result['mean_time_us']:.2f} us")
        bd = max_bw_result.get('phase_breakdown', {})
        comm_mean = bd.get('communication_latency_ns', {}).get('mean', 0)
        total_ns_mean = bd.get('total_ns', {}).get('mean', 0)
        print(f"  通信总时延: {comm_mean:.2f} ns")
        print(f"  总耗时: {total_ns_mean:.2f} ns")
        print(f"{'*' * 120}")

    # 带宽趋势分析
    print("\n" + "=" * 80)
    print("带宽趋势分析")
    print("=" * 80)

    print(f"\n{'数据大小':<12} {'Mean BW(GB/s)':<18} {'Max BW(GB/s)':<18} {'趋势'}")
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

    Args:
        results: 测试结果列表
        output_csv: CSV文件路径
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
            'sqe_assembly_ns_mean': bd.get('sqe_assembly_ns', {}).get('mean', 0),
            'p2p_send_ns_mean': bd.get('p2p_send_ns', {}).get('mean', 0),
            'communication_latency_ns_mean': bd.get('communication_latency_ns', {}).get('mean', 0),
            'setstatus_ns_mean': bd.get('setstatus_ns', {}).get('mean', 0),
            'waitdispatch_ns_mean': bd.get('waitdispatch_ns', {}).get('mean', 0),
            'local_copy_ns_mean': bd.get('local_copy_ns', {}).get('mean', 0),
            'total_ns_mean': bd.get('total_ns', {}).get('mean', 0),
            'total_us_mean': bd.get('total_us', {}).get('mean', 0),
            'mean_comm_latency_ns': r.get('mean_comm_latency_ns', 0),
            'mean_total_ns': r.get('mean_total_ns', 0),
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
        f.write(f"时延迭代次数: {args.latency_iterations}\n")
        f.write(f"预热次数: {args.warmup}\n")
        f.write(f"AIV核数: {aiv_num}\n")
        f.write(f"通信模式: SQE级HcclCombinOpParam URMA\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"时延单位: ns (1 cycle = 20ns)\n")
        f.write("\n")

        # 各测试结果
        for r in results:
            f.write(f"\n2. {r['test_type']} 测试结果\n")
            f.write("-" * 40 + "\n")
            f.write(f"数据大小: {r['data_size_label']} ({r['data_size_bytes']} bytes)\n")
            f.write(f"配置: bs={r.get('bs', 'N/A')}, h={r.get('h', 'N/A')}, max_bs={r.get('max_bs', 'N/A')}\n")
            f.write(f"迭代次数: {r.get('iterations', 'N/A')}\n")

            bd = r.get('phase_breakdown', {})
            if bd:
                phase_labels = {
                    'sqe_assembly_ns':         'SQE组装',
                    'p2p_send_ns':             'P2P数据传输',
                    'communication_latency_ns': '通信总时延',
                    'setstatus_ns':            'SetStatus',
                    'waitdispatch_ns':         'WaitDispatch',
                    'local_copy_ns':           'LocalWindowCopy',
                    'total_ns':                '总耗时(ns)',
                    'total_us':                '总耗时(us)',
                }
                for phase_name, label in phase_labels.items():
                    stats = bd.get(phase_name, {})
                    if stats:
                        unit = 'us' if phase_name == 'total_us' else 'ns'
                        f.write(f"  {label}: mean={stats.get('mean', 0):.2f} {unit}, "
                                f"min={stats.get('min', 0):.2f} {unit}, "
                                f"max={stats.get('max', 0):.2f} {unit}, "
                                f"std={stats.get('std', 0):.2f} {unit}, "
                                f"P50={stats.get('p50', 0):.2f} {unit}, "
                                f"P90={stats.get('p90', 0):.2f} {unit}, "
                                f"P99={stats.get('p99', 0):.2f} {unit}\n")

            if r.get('mean_bandwidth_gbps', 0) > 0:
                f.write(f"平均带宽: {r['mean_bandwidth_gbps']:.2f} GB/s\n")
                f.write(f"最大带宽: {r['max_bandwidth_gbps']:.2f} GB/s\n")

            if r.get('jitter'):
                jitter = r['jitter']
                f.write(f"\n抖动分析:\n")
                f.write(f"  平均抖动: {jitter['jitter_mean']:.2f} ns\n")
                f.write(f"  最大抖动: {jitter['jitter_max']:.2f} ns\n")
                f.write(f"  抖动标准差: {jitter['jitter_std']:.2f} ns\n")

            f.write("\n")

        # 汇总表格
        f.write("\n3. 汇总表格\n")
        f.write("-" * 100 + "\n")
        header = f"{'类型':<12} {'大小':<10} {'SQE(ns)':<12} {'P2P(ns)':<12} {'CommLat(ns)':<14} {'Total(ns)':<12} {'Total(us)':<12} {'Mean BW(GB/s)':<16} {'Max BW(GB/s)':<16}\n"
        f.write(header)
        f.write("-" * 100 + "\n")
        for r in results:
            bd = r.get('phase_breakdown', {})
            row = f"{r['test_type']:<12} {r['data_size_label']:<10} " \
                  f"{bd.get('sqe_assembly_ns', {}).get('mean', 0):<12.2f} " \
                  f"{bd.get('p2p_send_ns', {}).get('mean', 0):<12.2f} " \
                  f"{bd.get('communication_latency_ns', {}).get('mean', 0):<14.2f} " \
                  f"{bd.get('total_ns', {}).get('mean', 0):<12.2f} " \
                  f"{bd.get('total_us', {}).get('mean', 0):<12.2f} " \
                  f"{r.get('mean_bandwidth_gbps', 0):<16.2f} " \
                  f"{r.get('max_bandwidth_gbps', 0):<16.2f}\n"
            f.write(row)
        f.write("-" * 100 + "\n")

    print(f"测试报告已保存到: {report_path}")

# endregion


# region 结果汇总打印

def print_summary(results):
    """打印最终测试结果汇总"""
    print("\n" + "=" * 80)
    print("URMA P2P Benchmark 测试汇总")
    print("=" * 80)

    for r in results:
        test_type = r['test_type']
        bd = r.get('phase_breakdown', {})

        if test_type == 'overhead':
            print(f"\n[Overhead 测试 - {r['data_size_label']}]")
            comm_lat = bd.get('communication_latency_ns', {}).get('mean', 0)
            sqe = bd.get('sqe_assembly_ns', {}).get('mean', 0)
            p2p = bd.get('p2p_send_ns', {}).get('mean', 0)
            total_ns = bd.get('total_ns', {}).get('mean', 0)
            total_us = bd.get('total_us', {}).get('mean', 0)
            print(f"  通信总时延: {comm_lat:.2f} ns")
            print(f"  SQE组装: {sqe:.2f} ns")
            print(f"  P2P传输: {p2p:.2f} ns")
            print(f"  总耗时: {total_ns:.2f} ns ({total_us:.2f} us)")

        elif test_type == 'latency':
            print(f"\n[Latency 测试 - {r['data_size_label']}]")
            comm_lat = bd.get('communication_latency_ns', {})
            if comm_lat:
                print(f"  通信总时延: mean={comm_lat['mean']:.2f} ns, "
                      f"P50={comm_lat['p50']:.2f} ns, "
                      f"P90={comm_lat['p90']:.2f} ns, "
                      f"P99={comm_lat['p99']:.2f} ns")
            sqe = bd.get('sqe_assembly_ns', {}).get('mean', 0)
            p2p = bd.get('p2p_send_ns', {}).get('mean', 0)
            print(f"  SQE组装: {sqe:.2f} ns")
            print(f"  P2P传输: {p2p:.2f} ns")
            if r.get('jitter'):
                print(f"  平均抖动: {r['jitter']['jitter_mean']:.2f} ns")

        elif test_type == 'bandwidth':
            bw = r.get('mean_bandwidth_gbps', 0)
            max_bw = r.get('max_bandwidth_gbps', 0)
            if bw > 0:
                print(f"\n[Bandwidth 测试 - {r['data_size_label']}]")
                print(f"  平均带宽: {bw:.2f} GB/s")
                print(f"  最大带宽: {max_bw:.2f} GB/s")
                print(f"  平均时间: {r.get('mean_time_us', 0):.2f} us")
                total_ns = bd.get('total_ns', {}).get('mean', 0)
                comm_lat = bd.get('communication_latency_ns', {}).get('mean', 0)
                print(f"  通信总时延: {comm_lat:.2f} ns")
                print(f"  总耗时: {total_ns:.2f} ns")

    # 找出最大带宽
    bandwidth_results = [r for r in results if r['test_type'] == 'bandwidth' and r.get('max_bandwidth_gbps', 0) > 0]
    if bandwidth_results:
        best = max(bandwidth_results, key=lambda x: x['max_bandwidth_gbps'])
        print(f"\n{'*' * 60}")
        print(f"峰值P2P带宽: {best['max_bandwidth_gbps']:.2f} GB/s (数据大小: {best['data_size_label']})")
        print(f"{'*' * 60}")

    # 找出最小overhead
    overhead_results = [r for r in results if r['test_type'] == 'overhead']
    if overhead_results:
        smallest = min(overhead_results, key=lambda x: x.get('mean_comm_latency_ns', float('inf')))
        print(f"\n最小通信总时延: {smallest.get('mean_comm_latency_ns', 0):.2f} ns "
              f"(数据大小: {smallest['data_size_label']})")

    print("\n" + "=" * 80)

# endregion


# region 主函数

def main():
    args = parse_args()
    setup_environment(args)

    results = []

    if args.test_type in ['overhead', 'all']:
        results.extend(run_overhead_test(args))
    if args.test_type in ['latency', 'all']:
        results.extend(run_latency_test(args))
    if args.test_type in ['bandwidth', 'all']:
        results.extend(run_bandwidth_test(args))

    export_csv(results, args.output_csv)
    generate_report(results, args.output_dir, args)
    print_summary(results)

    print(f"\n测试完成! 共执行 {len(results)} 个测试配置")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()