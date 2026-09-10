import copy
import csv
import datetime
import multiprocessing as mp
import os
import socket
import time
from enum import Enum
from math import ceil
from typing import List
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore
import torch.distributed as dist  # type: ignore
import torch.multiprocessing as mp  # type: ignore
import torch_npu  # type: ignore
import torchair
from torch.distributed import ReduceOp  # type: ignore
from torch.multiprocessing import Manager, Process  # type: ignore
from torchair.configs.compiler_config import CompilerConfig
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn
import cann_ops_transformer
torch_npu.npu.config.allow_internal_format = True
# 设置环境变量
os.environ['HCCL_BUFFSIZE'] = '2000'
# 全局配置
world_size = 4
server_num = 1
server_index = 0
master_ip = "127.0.0.1"
# 图模式配置
graph_type = 0  # 0: 单算子模式, 1: 图模式
is_full_core = True
aic_core_num = 32
aiv_core_num = 64
# shape 配置（matmul_allto_all_v2：M 为每 rank 的完整 M，不切分；N 按 world_size 切分）
# 批量执行：测试 shape = m_list × kn_list 笛卡尔积，每组生成独立 Profiling 子目录
# {profiling_path}_m{m}_k{k}_n{n}
# 约束：ceil(k/64) 须为偶数；n 须被 world_size 整除
m_list = [2048]
kn_list = [(2048, 3072), (3584, 3072)]  # (K, N) 组合列表
# profiling 配置
loop_time = 50
repeat_time = 1
warmup_time = 3
profiling_path = f"./Profiling_Result_mm_alltoall"  # 各 shape 会生成 {profiling_path}_m{m}_k{k}_n{n} 子目录
# 算子入参枚举（与 aclnn 定义一致）
Y_DTYPE_BF16 = 15
MX_QUANT_MODE = 6
SCALE_DTYPE_FP8_E8M0 = 293
MX_GROUP_SIZES = [1, 1, 32]
def npu_prefetch(open: bool, weight, depend, size, offset):
    if open:
        return torch_npu.npu_prefetch(weight, depend, size, offset)
    else:
        return 0
def set_device(rank):
    torch.npu.set_device(rank % world_size)
    print(f"[INFO] server_index: {server_index}, rank: {rank}, device: {torch.npu.current_device()}")
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
def init_hccl_comm(rank):
    print(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50006')
    print(f"[INFO] device_{rank} HCCL通信链路创建成功")
    global_group = dist.new_group(backend='hccl', ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} 全局通信组创建成功")
    ep_hcomm_info = global_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    return ep_hcomm_info, global_group
def run_mmalltoall_profile(queue, rank):
    group = None
    try:
        set_device(rank)
        ep_hcomm_info, group = init_hccl_comm(rank)
        device = f"npu:{rank}"
        tensor_192mb = torch.rand((49152, 1), dtype=torch.float32).npu()
        tensor = torch.ones(10000, 10000).to(torch.float16).npu()
        for kn_idx, (k, n) in enumerate(kn_list):
            print(f"[INFO] rank {rank} 开始测试 (k,n)=({k},{n}) "
                  f"({kn_idx + 1}/{len(kn_list)})", flush=True)
            # 与 (k,n) 相关、与 m 无关的输入，每组 (k,n) 重新分配
            # x2: 底层 [N, K] 连续存储，取转置视图 [K, N] 传入（transpose_x2 恒为 true）
            x2_weight = torch.randn(
                n, k,
                dtype=torch.float32
            ).to(torch.float8_e4m3fn).npu()
            x2 = x2_weight.t()
            x2_scale = torch.full((n, ceil(k / 64), 2), 127, dtype=torch.uint8).npu()
            for m_idx, m in enumerate(m_list):
                print(f"[INFO] rank {rank} 开始测试 m={m} "
                      f"({m_idx + 1}/{len(m_list)})", flush=True)
                # 与 m 相关的输入，每组 shape 重新分配
                x1 = torch.randn(
                    m, k,
                    dtype=torch.float32
                ).to(torch.float8_e4m3fn).npu()
                x1_scale = torch.full((m, ceil(k / 64), 2), 127, dtype=torch.uint8).npu()
                prof_path = f"{profiling_path}_m{m}_k{k}_n{n}"
                with get_profiling(loop_time, repeat_time, warmup_time, prof_path) as prof:
                    for i in range(loop_time):
                        scale_l2 = torch.max(tensor_192mb)
                        y = cann_ops_transformer.quant_matmul_all_to_all(
                            x1=x1,
                            x2=x2,
                            group=group,
                            bias=None,
                            x1_scale=x1_scale,
                            x2_scale=x2_scale,
                            y_dtype=Y_DTYPE_BF16,
                            x1_quant_mode=MX_QUANT_MODE,
                            x2_quant_mode=MX_QUANT_MODE,
                            group_sizes=MX_GROUP_SIZES,
                            x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                            x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
                            comm_mode="urma")
                        if i == 1:
                            for _ in range(100):
                                torch.exp(tensor)
                        # 每次调用推进一个 step：Step Id 0~loop-1 与调用次数一一对应，
                        # 其中 step 0 为 profiler warmup，1~loop-1 为 active 采集段
                        prof.step()
                torch.npu.synchronize()
                # 组间同步，确保所有 rank 完成当前 shape 再进入下一组
                dist.barrier(group=group)
                del x1, x1_scale, y
                torch.npu.empty_cache()
            del x2_weight, x2, x2_scale
            torch.npu.empty_cache()
        dist.destroy_process_group()
        # ✅ 第一步：所有进程同步，确保没有进程提前退出
        print(f"[INFO] rank {rank} 所有进程同步完成", flush=True)
        # ✅ 第二步：同步后再向队列放入结果
        queue.put([rank, "success"])
        print(f"[INFO] rank {rank} 结果已放入队列", flush=True)
    except Exception as e:
        print(f"[ERROR] rank {rank} 执行失败: {str(e)}", flush=True)
        # 即使失败，也要尝试让所有进程同步，避免死锁
        try:
            dist.barrier(group=group, timeout=5)
        except:
            pass
        queue.put([rank, f"error: {str(e)}"])

    finally:
        # ✅ 第三步：先销毁通信组（集体操作），再清理NPU资源
        if dist.is_initialized():
            print(f"[INFO] rank {rank} 开始销毁HCCL通信组", flush=True)
            dist.destroy_process_group(group)
            print(f"[INFO] rank {rank} HCCL通信组销毁成功", flush=True)

        # ✅ 第四步：最后再清理NPU缓存
        torch.npu.empty_cache()
        print(f"[INFO] rank {rank} 资源清理完成", flush=True)
def main():
    print(f"[INFO] 开始运行 matmul_allto_all_v2 性能测试，world_size={world_size}")

    # 多进程，每个进程调用一张卡执行目标函数
    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method('forkserver', force=True)

    for rank in range(world_size):
        proc = Process(target=run_mmalltoall_profile, args=(result_queue, rank))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    # 收集结果
    rank_outputs = [None] * world_size
    print(f"[INFO] 所有进程执行完成，收集到 {len(rank_outputs)} 个结果")
if __name__ == "__main__":
    main()
