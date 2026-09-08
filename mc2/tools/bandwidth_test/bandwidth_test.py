#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file bandwidth_test.py
#  \brief

import os
import socket
import logging
import shutil
from pathlib import Path

from typing import List, Tuple, Dict, Any

script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "torch_extensions_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
import pandas as pd
import numpy as np

from cann_ops_transformer.ops import BandwidthTestBuffer, bandwidth_test

if __package__:
    from .config import CONFIG
else:
    from config import CONFIG

logging.basicConfig(
    level=CONFIG.diagnostics.log_level,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

URMA_MAX_BATCH_TOKENS = 2048  # Matches the current tiling cap for all supported shapes.

# Resolve the environment and derive runtime values from config.py.
server_index = CONFIG.parallel.server_index
master_ip = CONFIG.parallel.master_ip
if master_ip is None:
    master_ip = "71.20.45.123" if CONFIG.parallel.server_num > 1 else "127.0.0.1"
if CONFIG.parallel.is_cloud:
    task_index = os.environ.get("VC_TASK_INDEX")
    if task_index is None:
        raise EnvironmentError("环境变量 VC_TASK_INDEX 未设置，请检查云环境配置")
    server_index = int(task_index)
    host_urls = os.environ.get("VC_WORKER_HOSTS")
    if host_urls is None or host_urls == "":
        raise EnvironmentError("环境变量 VC_WORKER_HOSTS 未设置，请检查云环境配置")
    host_url_list = host_urls.split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

aic_core_num = 32 if CONFIG.parallel.is_full_core else CONFIG.parallel.partial_core_num
aiv_core_num = aic_core_num * 2
aiv_num = aiv_core_num if CONFIG.parallel.aiv_num is None else CONFIG.parallel.aiv_num
world_size = CONFIG.parallel.server_num * CONFIG.parallel.rank_per_dev
ranks_list = list(range(world_size))
test_configs = (
    [{"bs": CONFIG.shape.bs, "data_size": CONFIG.shape.data_size, "max_bs": CONFIG.shape.max_bs}]
    if CONFIG.run.is_single_test
    else CONFIG.shape.test_configs
)
input_dtype = torch.bfloat16 if CONFIG.shape.x_dtype == "bf16" else torch.float16
dtype_bytes = torch.empty(0, dtype=input_dtype).element_size()
mode = CONFIG.model.mode
if mode is None:
    mode = 0 if CONFIG.run.is_precision_test else 1
if mode not in (0, 1, 3):
    raise ValueError("mode must be 0, 1 or 3")
if CONFIG.model.shared_jetty:
    if CONFIG.model.comm_alg != "urma" or not 1 <= CONFIG.model.shared_jetty_num <= 56:
        raise ValueError("shared_jetty requires urma and shared_jetty_num in [1, 56]")
    if CONFIG.model.shared_jetty_comm_level not in (0, 1, 2):
        raise ValueError("shared_jetty_comm_level must be 0, 1 or 2")
    if aiv_num != -1 and CONFIG.model.shared_jetty_num > aiv_num:
        raise ValueError("shared_jetty_num must not exceed aiv_num")
if mode == 3 and CONFIG.model.comm_alg != "urma":
    raise ValueError("mode=3 requires comm_alg=urma")
if CONFIG.run.is_precision_test and mode != 0:
    raise ValueError("precision testing requires mode=0")


def cal_hccl_buffsize(test_max_bs, h, ep_world_size):
    # A5 allocates twice HCCL_BUFFSIZE; each half includes one reserved token block.
    row_bytes = ((h * 2 + 511) // 512) * 512
    required_bytes = 1024 * 1024 + 2 * (ep_world_size + 1) * test_max_bs * row_bytes
    if mode == 3:
        required_bytes = 1024 * 1024 + 2 * test_max_bs * h * dtype_bytes
    return str((required_bytes + 2 * 1024 * 1024 - 1) // (2 * 1024 * 1024))


if not test_configs:
    raise ValueError("CONFIG.shape.test_configs must not be empty")
os.environ["HCCL_BUFFSIZE"] = str(
    max(
        int(cal_hccl_buffsize(cfg["max_bs"], cfg["data_size"], world_size))
        for cfg in test_configs
    )
)


def save_tensor_to_bin(tensor, bin_name):
    os.makedirs(CONFIG.diagnostics.tensor_output_path, exist_ok=True)
    tensor.cpu().numpy().tofile(
        os.path.join(CONFIG.diagnostics.tensor_output_path, f"{bin_name}.bin")
    )


def print_setting():
    logger.info(f"{CONFIG.parallel.server_num=}")
    logger.info(f"{CONFIG.parallel.rank_per_dev=}")
    logger.info(f"{world_size=}")
    logger.info(f"{mode=}")
    logger.info(f"{CONFIG.model.comm_alg=}")
    logger.info(f"{CONFIG.model.batch_tokens=}")
    logger.info(
        f"{CONFIG.model.shared_jetty=}, {CONFIG.model.shared_jetty_num=}, "
        f"{CONFIG.model.shared_jetty_comm_level=}"
    )
    logger.info(f"{CONFIG.model.graph_type=}")
    logger.info(f"{aiv_num=}")
    logger.info(f"{CONFIG.run.is_precision_test=}")
    logger.info(f"{CONFIG.run.is_bandwidth_test=}")
    logger.info(f"{test_configs=}")


def check_setting():
    def error_log_and_exit(log_info, exit_code=1):
        logger.error(log_info)
        raise RuntimeError(f"ERROR: {log_info}")

    for i, cfg in enumerate(test_configs):
        if cfg["bs"] < 1:
            error_log_and_exit(f"test_configs[{i}]: bs 应大于等于 1")
        if cfg["data_size"] < 1:
            error_log_and_exit(f"test_configs[{i}]: data_size 应大于等于 1")
        if cfg["max_bs"] < cfg["bs"]:
            error_log_and_exit(f"test_configs[{i}]: max_bs 应大于等于 bs")

    if (server_index < 0) or (server_index >= CONFIG.parallel.server_num):
        error_log_and_exit("server_index 取值范围为 [0, server_num)")

    if CONFIG.parallel.server_num * CONFIG.parallel.rank_per_dev != world_size:
        error_log_and_exit("server_num * rank_per_dev 必须等于 world_size")

    if (CONFIG.model.graph_type == 1) or (CONFIG.model.graph_type == 2):
        error_log_and_exit("当前不支持动态图模式, graph_type 合法值仅有 0 和 3")


# endregion

# region 性能采集函数


def get_profiling(num_loop, num_repeat, num_warmup, profiling_output_path):
    active = num_loop // num_repeat - num_warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=num_warmup, active=active, repeat=num_repeat, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profiling_output_path
        ),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
    return prof


def clean_previous_profiling_result():
    # Only remove results belonging to the configured profiling path.
    result_path = Path(CONFIG.run.profiling_path).resolve()
    working_path = Path.cwd().resolve()
    if result_path == working_path or result_path in working_path.parents:
        raise ValueError("profiling_path must not be the working directory or its ancestor")
    paths = [result_path]
    if result_path.parent.exists():
        paths.extend(
            path
            for path in result_path.parent.iterdir()
            if path.name.startswith(f"{result_path.name}_config_")
            and path.name[len(result_path.name) + len("_config_") :].isdigit()
        )
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)


def get_bandwidth_performance(
    prof_result_path: str, data_size_bytes: int, num_world_size: int, test_bs: int
):
    """从 profiling 结果中提取性能数据并计算带宽"""
    execution_times = []

    for dirpath, _, filenames in os.walk(prof_result_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                bandwidth_test_info = op_summary[
                    op_summary["OP Type"] == "BandwidthTest"
                ]

                if not bandwidth_test_info.empty:
                    op_execute_time_series = bandwidth_test_info[
                        "Task Duration(us)"
                    ].tail(CONFIG.run.performance_sample_count)
                    execution_times.extend(op_execute_time_series.tolist())

    if not execution_times:
        logger.warning("No performance data found")
        return None

    execution_times_us = np.array(execution_times)
    mean_time_us = np.mean(execution_times_us)
    min_time_us = np.min(execution_times_us)
    max_time_us = np.max(execution_times_us)

    mean_time_s = mean_time_us / 1e6
    min_time_s = min_time_us / 1e6

    single_rank_data_bytes = (
        test_bs * data_size_bytes * ((num_world_size - 1) / num_world_size)
    )
    if mode == 3:
        single_rank_data_bytes = (test_bs // num_world_size) * (num_world_size - 1) * data_size_bytes
    total_data_bytes = single_rank_data_bytes * num_world_size
    mean_bandwidth_gbps = (
        (total_data_bytes / mean_time_s) / 1024 / 1024 / 1024 / num_world_size
    )
    max_bandwidth_gbps = (
        (total_data_bytes / min_time_s) / 1024 / 1024 / 1024 / num_world_size
    )
    single_rank_data_kb = single_rank_data_bytes / 1024

    logger.info("=" * 60)
    logger.info("Bandwidth Test Performance Results")
    logger.info("=" * 60)
    logger.info(f"Total data per iteration: {total_data_bytes / 1e6:.2f} MB")
    logger.info(f"World size: {num_world_size}")
    logger.info(f"BS per rank: {test_bs}")
    logger.info(f"Data size (H): {data_size_bytes // dtype_bytes}")
    logger.info("-" * 60)
    logger.info(f"Mean execution time: {mean_time_us:.2f} us")
    logger.info(f"Min execution time:  {min_time_us:.2f} us")
    logger.info("-" * 60)
    logger.info(f"Mean bandwidth: {mean_bandwidth_gbps:.2f} GB/s")
    logger.info(f"Max bandwidth:  {max_bandwidth_gbps:.2f} GB/s")
    logger.info("=" * 60)

    return {
        "mean_time_us": mean_time_us,
        "min_time_us": min_time_us,
        "max_time_us": max_time_us,
        "mean_bandwidth_gbps": mean_bandwidth_gbps,
        "max_bandwidth_gbps": max_bandwidth_gbps,
        "total_data_bytes": total_data_bytes,
        "single_rank_data_kb": single_rank_data_kb,
        "data_size": data_size_bytes // dtype_bytes,
        "bs": test_bs,
    }


# endregion

# region 图模式相关


def compile_model(model):
    if CONFIG.model.graph_type == 0:
        raise RuntimeError("单算子模式下不应调用compile_model函数")

    logger.info(f"------------------graph_type: {CONFIG.model.graph_type}------------------")
    import torchair

    from torchair.configs.compiler_config import CompilerConfig

    if CONFIG.parallel.is_full_core:
        compiler_config = None
    else:
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
        logger.info(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)

    if CONFIG.model.graph_type == 3:
        logger.info("graph_type=3, 测试场景: aclgraph入图模式")
        os.makedirs(CONFIG.model.static_kernel_path, exist_ok=True)
        compiler_config = CompilerConfig()
        logger.info("启用aclnn静态内核（aclgraph），当前只支持全核")
        compiler_config.mode = "reduce-overhead"
        # 开启静态Kernel编译
        compiler_config.debug.aclgraph.clone_input = False
        compiler_config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        compiler_config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = (
            CONFIG.model.static_kernel_path
        )

        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        compiled_model = torch.compile(
            model, backend=npu_backend, dynamic=False, fullgraph=True
        )

    return compiled_model


class BANDWIDTH_TEST_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        dstrank_id,
        sym_buffer,
        test_world_size,
        test_max_bs,
        test_mode,
        test_comm_alg,
        test_aiv_num,
        test_batch_tokens,
        test_shared_jetty,
        test_shared_jetty_num,
    ):
        local_loop = CONFIG.run.loop if CONFIG.run.is_bandwidth_test else 1
        output = None
        for _ in range(local_loop):
            output = bandwidth_test(
                x=x,
                dstrank_id=dstrank_id,
                sym_buffer=sym_buffer,
                world_size=test_world_size,
                max_bs=test_max_bs,
                mode=test_mode,
                comm_alg=test_comm_alg,
                aiv_num=test_aiv_num,
                batch_tokens=test_batch_tokens,
                shared_jetty=test_shared_jetty,
                shared_jetty_num=test_shared_jetty_num,
            )
        return output


# endregion

# region 通信初始化


def set_device(rank):
    torch_npu.npu.set_device(rank % CONFIG.parallel.rank_per_dev)
    logger.info(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank, world_size_local):
    logger.info(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size_local,
        init_method=f"tcp://{master_ip}:{CONFIG.parallel.master_port}",
    )
    logger.info(f"device_{rank} init_process_group success")

    logger.info(f"device {rank} 初始化通信域")
    comm_group = dist.new_group(backend="hccl", ranks=ranks_list)

    logger.debug("[DEBUG] device_%s 等待所有进程初始化完成...", rank)
    dist.barrier()
    logger.debug("[DEBUG] device_%s barrier完成", rank)

    torch.npu.synchronize()
    logger.debug("[DEBUG] device_%s NPU同步完成", rank)

    group_name = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    logger.debug(
        "[DEBUG] device_%s HCCL通信域名称: %s, 长度: %s",
        rank,
        group_name,
        len(group_name) if group_name else 0,
    )

    if not group_name or len(group_name) == 0:
        raise RuntimeError(f"device_{rank} HCCL通信域获取失败")

    dist.barrier()
    torch.npu.synchronize()
    logger.debug("[DEBUG] device_%s 通信域完全就绪", rank)

    return BandwidthTestBuffer(comm_group, comm_alg=CONFIG.model.comm_alg,
                               shared_jetty=CONFIG.model.shared_jetty,
                               shared_jetty_num=CONFIG.model.shared_jetty_num,
                               shared_jetty_comm_level=CONFIG.model.shared_jetty_comm_level), comm_group


# endregion

# region 输入生成


def generate_inputs(
    test_bs: int, test_data_size: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """生成测试输入数据，每个发送方内部的目标rank均衡"""
    if mode == 3:
        # Only tensor shapes are used by this mode; the registered Win supplies the payload.
        return ([torch.empty((test_bs, test_data_size), dtype=input_dtype) for _ in range(world_size)],
                [torch.empty((test_bs,), dtype=torch.int32) for _ in range(world_size)])
    torch.manual_seed(CONFIG.shape.random_seed)

    x_per_rank = [
        torch.empty([test_bs, test_data_size], dtype=input_dtype).uniform_(-1, 1)
        for _ in range(world_size)
    ]

    dstrank_id_per_rank = []
    for _ in range(world_size):
        # 计算每个目标 rank 应分配的数量（均衡分配）
        base = test_bs // world_size
        rem = test_bs % world_size
        targets = []
        for r in range(world_size):
            count = base + (1 if r < rem else 0)
            targets.extend([r] * count)
        # 随机打乱顺序，使每个发送方内部的目标顺序随机
        indices = torch.randperm(test_bs).tolist()
        shuffled_targets = [targets[i] for i in indices]
        dstrank_id_per_rank.append(torch.tensor(shuffled_targets, dtype=torch.int32))

    return x_per_rank, dstrank_id_per_rank


def prepare_npu_inputs(x, dstrank_id, test_mode):
    if test_mode == 3:
        return (torch.empty(x.shape, dtype=input_dtype, device="npu"),
                torch.empty(dstrank_id.shape, dtype=torch.int32, device="npu"))
    return x.to(input_dtype).npu(), dstrank_id.to(torch.int32).npu()


# endregion

# region Golden 生成


def gen_golden_data(
    rank: int,
    x_per_rank: List[torch.Tensor],
    dstrank_id_per_rank: List[torch.Tensor],
    test_max_bs: int,
    test_data_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 golden 参考数据"""
    y = x_per_rank[0].new_zeros([test_max_bs * world_size, test_data_size])
    receive_cnt = torch.zeros([world_size], dtype=torch.int32)

    offset = 0
    for src_rank in range(world_size):
        x = x_per_rank[src_rank]
        dstrank_id = dstrank_id_per_rank[src_rank]

        dst_rank = dstrank_id % world_size

        mask = dst_rank == rank
        tokens_for_me = x[mask]
        count = mask.sum().item()

        if count > 0:
            y[offset : offset + count] = tokens_for_me
            offset += count

        receive_cnt[src_rank] = count

    return y, receive_cnt


# endregion

# region 精度对比


def compare_result(
    npu_output,
    golden_output,
    name: str,
    rtol: float = CONFIG.diagnostics.precision_rtol,
    atol: float = CONFIG.diagnostics.precision_atol,
    verbose: bool = CONFIG.diagnostics.precision_verbose,
) -> bool:
    """对比 NPU 输出与 Golden 输出"""
    if isinstance(npu_output, (list, tuple)):
        all_pass = True
        for i, (npu_out, golden_out) in enumerate(zip(npu_output, golden_output)):
            if not compare_result(
                npu_out, golden_out, f"{name}_{i}", rtol, atol, verbose
            ):
                all_pass = False
        return all_pass

    logger.info(f"{'=' * 60}")
    logger.info(f"对比 {name}")
    logger.info(f"NPU shape: {npu_output.shape}, Golden shape: {golden_output.shape}")
    logger.info(f"{'=' * 60}")

    if verbose:
        logger.info("NPU 输出:")
        logger.info(npu_output)
        logger.info("Golden 输出:")
        logger.info(golden_output)

    if torch.equal(npu_output, golden_output):
        logger.info(f"[PASS] {name} 精度对比通过 (完全一致)")
        return True

    diff = torch.abs(npu_output - golden_output)
    max_diff = diff.max().item()

    if torch.is_floating_point(npu_output):
        mean_diff = diff.mean().item()
        rel_diff = diff / (torch.abs(golden_output) + 1e-8)
        max_rel_diff = rel_diff.max().item()

        if verbose:
            logger.info("差异:")
            logger.info(diff)
            logger.info("相对差异:")
            logger.info(rel_diff)

        if max_rel_diff > rtol or max_diff > atol:
            logger.error(f"[ERROR] {name} 精度不满足要求:")
            logger.error(f"  最大相对误差: {max_rel_diff}")
            logger.error(f"  容忍范围: rtol={rtol}, atol={atol}")
            return False
        else:
            logger.info(f"[PASS] {name} 精度对比通过 (误差在容忍范围内)")
            logger.info(f"  最大绝对误差: {max_diff}, 最大相对误差: {max_rel_diff:.2e}")
    else:
        mean_diff = diff.float().mean().item()

        if verbose:
            logger.info("差异:")
            logger.info(diff)

        if max_diff > atol:
            logger.error(f"[ERROR] {name} 精度不满足要求:")
            logger.error(f"  容忍范围: atol={atol}")
            return False
        else:
            logger.info(f"[PASS] {name} 精度对比通过 (误差在容忍范围内)")
            logger.info(f"  最大绝对误差: {max_diff}")

    return True


# endregion

# region NPU 执行


def run_bandwidth_test_npu(
    queue,
    rank,
    x,
    dstrank_id,
    sym_buffer,
    test_max_bs=None,
    test_mode=None,
    test_world_size=None,
    is_perf_mode=False,
    destroy_comm=True,
):
    """执行 bandwidth_test 算子"""
    set_device(rank)

    if sym_buffer is None:
        sym_buffer, _ = init_hccl_comm(rank, test_world_size)

    if CONFIG.diagnostics.is_save_tensor_to_bin and test_mode != 3:
        save_tensor_to_bin(x.to(torch.float32), "x_in_" + str(rank))
        save_tensor_to_bin(dstrank_id.to(torch.int32), "dstrank_id_" + str(rank))

    logger.info(f"[INFO] device_{rank} 构造bandwidth_test算子输入数据")

    x_npu, dstrank_id_npu = prepare_npu_inputs(x, dstrank_id, test_mode)
    torch.npu.synchronize()
    logger.debug("[DEBUG] device_%s 开始调用算子", rank)

    # 根据模式执行
    if CONFIG.model.graph_type == 0:
        bandwidth_test_kwargs = {
            "x": x_npu,
            "dstrank_id": dstrank_id_npu,
            "sym_buffer": sym_buffer,
            "world_size": test_world_size,
            "max_bs": test_max_bs,
            "mode": test_mode,
            "comm_alg": CONFIG.model.comm_alg,
            "aiv_num": aiv_num,
            "batch_tokens": CONFIG.model.batch_tokens,
            "shared_jetty": CONFIG.model.shared_jetty,
            "shared_jetty_num": CONFIG.model.shared_jetty_num,
        }

        if not is_perf_mode:
            y, receive_cnt = bandwidth_test(
                **bandwidth_test_kwargs
            )
        else:
            with get_profiling(
                CONFIG.run.loop, CONFIG.run.repeat, CONFIG.run.warmup, CONFIG.run.profiling_path
            ) as prof:
                for i in range(CONFIG.run.loop):
                    torch.npu.synchronize()
                    y, receive_cnt = bandwidth_test(
                        **bandwidth_test_kwargs
                    )
                    prof.step()
    else:
        bandwidth_test_kwargs = {
            "x": x_npu,
            "dstrank_id": dstrank_id_npu,
            "sym_buffer": sym_buffer,
            "test_world_size": test_world_size,
            "test_max_bs": test_max_bs,
            "test_mode": test_mode,
            "test_comm_alg": CONFIG.model.comm_alg,
            "test_aiv_num": aiv_num,
            "test_batch_tokens": CONFIG.model.batch_tokens,
            "test_shared_jetty": CONFIG.model.shared_jetty,
            "test_shared_jetty_num": CONFIG.model.shared_jetty_num,
        }
        if not is_perf_mode:
            model = compile_model(BANDWIDTH_TEST_GRAPH_Model())
            y, receive_cnt = model(**bandwidth_test_kwargs)
        else:
            torch._dynamo.reset()
            model = compile_model(BANDWIDTH_TEST_GRAPH_Model())
            with get_profiling(
                CONFIG.run.graph_loop,
                CONFIG.run.graph_repeat,
                CONFIG.run.graph_warmup,
                CONFIG.run.profiling_path,
            ) as prof:
                for i in range(CONFIG.run.graph_loop):
                    y, receive_cnt = model(**bandwidth_test_kwargs)
                    prof.step()

    torch.npu.synchronize()
    torch.npu.empty_cache()

    if destroy_comm:
        dist.destroy_process_group()

    logger.info(f"rank {rank} npu finished! \n")

    if CONFIG.diagnostics.is_save_tensor_to_bin and test_mode != 3:
        save_tensor_to_bin(y.to(torch.float32), "y_out_" + str(rank))
        save_tensor_to_bin(receive_cnt, "receive_cnt_" + str(rank))

    outputs = [None, None] if test_mode == 3 else [y.cpu(), receive_cnt.cpu()]
    queue.put([rank, outputs])


def run_bandwidth_test_npu_bandwidth(
    queue, rank, x, dstrank_id, test_max_bs, test_mode, test_world_size
):
    """专门用于带宽测试的执行函数"""
    run_bandwidth_test_npu(
        queue,
        rank,
        x,
        dstrank_id,
        None,
        test_max_bs,
        test_mode,
        test_world_size,
        is_perf_mode=True,
        destroy_comm=True,
    )


def run_bandwidth_test_npu_multi_config(queue, rank, configs):
    """多配置带宽测试，共享通信域

    Args:
        queue: 多进程通信队列
        rank: 当前进程rank
        configs: 配置列表，每个元素为 (x, dstrank_id, test_max_bs, test_data_size, config_index) 元组
    """
    set_device(rank)

    if not configs:
        queue.put([rank, []])
        return

    _, _, _, _, test_world_size, _ = configs[0]

    logger.info(f"[INFO] device_{rank} 创建通信域（多配置共享）")
    sym_buffer, _ = init_hccl_comm(rank, test_world_size)

    results = []

    for _, (
        x,
        dstrank_id,
        test_max_bs,
        test_data_size,
        test_world_size,
        config_index,
    ) in enumerate(configs):
        test_bs = x.size(0)
        logger.info(f"{'=' * 60}")
        logger.info(
            f"[Rank {rank}] Config [{config_index}] bs={test_bs}, data_size={test_data_size},"
            f"max_bs={test_max_bs}"
        )
        logger.info(f"{'=' * 60}")

        logger.info(f"[INFO] device_{rank} 构造bandwidth_test算子输入数据")

        x_npu, dstrank_id_npu = prepare_npu_inputs(x, dstrank_id, mode)

        torch.npu.synchronize()
        logger.debug("[DEBUG] device_%s 开始调用算子", rank)
        dist.barrier()

        config_profiling_path = f"{CONFIG.run.profiling_path}_config_{config_index}"

        if CONFIG.model.graph_type == 0:
            bandwidth_test_kwargs = {
                "x": x_npu,
                "dstrank_id": dstrank_id_npu,
                "sym_buffer": sym_buffer,
                "world_size": test_world_size,
                "max_bs": test_max_bs,
                "mode": mode,
                "comm_alg": CONFIG.model.comm_alg,
                "aiv_num": aiv_num,
                "batch_tokens": CONFIG.model.batch_tokens,
                "shared_jetty": CONFIG.model.shared_jetty,
                "shared_jetty_num": CONFIG.model.shared_jetty_num,
            }
            with get_profiling(
                CONFIG.run.loop, CONFIG.run.repeat, CONFIG.run.warmup, config_profiling_path
            ) as prof:
                for j in range(CONFIG.run.loop):
                    torch.npu.synchronize()
                    y, receive_cnt = bandwidth_test(
                        **bandwidth_test_kwargs
                    )
                    prof.step()
        else:
            bandwidth_test_kwargs = {
                "x": x_npu,
                "dstrank_id": dstrank_id_npu,
                "sym_buffer": sym_buffer,
                "test_world_size": test_world_size,
                "test_max_bs": test_max_bs,
                "test_mode": mode,
                "test_comm_alg": CONFIG.model.comm_alg,
                "test_aiv_num": aiv_num,
                "test_batch_tokens": CONFIG.model.batch_tokens,
                "test_shared_jetty": CONFIG.model.shared_jetty,
                "test_shared_jetty_num": CONFIG.model.shared_jetty_num,
            }
            torch._dynamo.reset()
            model = compile_model(BANDWIDTH_TEST_GRAPH_Model())
            with get_profiling(
                CONFIG.run.graph_loop,
                CONFIG.run.graph_repeat,
                CONFIG.run.graph_warmup,
                config_profiling_path,
            ) as prof:
                for i in range(CONFIG.run.graph_loop):
                    y, receive_cnt = model(**bandwidth_test_kwargs)
                    prof.step()

        dist.barrier()
        torch.npu.synchronize()
        torch.npu.empty_cache()

        if CONFIG.diagnostics.is_save_tensor_to_bin and mode != 3:
            save_tensor_to_bin(
                y.to(torch.float32), f"y_out_{rank}_config{config_index}"
            )
            save_tensor_to_bin(receive_cnt, f"receive_cnt_{rank}_config{config_index}")

        results.append(
            {
                "config_index": config_index,
                "y": None if mode == 3 else y.cpu(),
                "receive_cnt": None if mode == 3 else receive_cnt.cpu(),
                "test_bs": test_bs,
                "test_data_size": test_data_size,
            }
        )

        logger.info(f"[INFO] device_{rank} Config [{config_index}] 完成")

    dist.destroy_process_group()

    queue.put([rank, results])


# endregion

# region 主测试函数


def run_precision_test(
    x_per_rank: List[torch.Tensor],
    dstrank_id_per_rank: List[torch.Tensor],
    test_max_bs: int,
    test_data_size: int,
) -> bool:
    """运行精度测试"""
    logger.info("=" * 60)
    logger.info("Precision Test")
    logger.info("=" * 60)

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        p = Process(
            target=run_bandwidth_test_npu,
            args=(
                result_queue,
                rank,
                x_per_rank[rank],
                dstrank_id_per_rank[rank],
                None,
                test_max_bs,
                mode,
                world_size,
                False,
                True,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    npu_outputs = {}

    if result_queue.qsize() == world_size:
        while not result_queue.empty():
            rank, output = result_queue.get()
            if output is not None:
                npu_outputs[rank] = output
            else:
                logger.error(f"[ERROR] Rank {rank} 执行失败")
                return False
    else:
        logger.error("Get result_queue Failed")
        return False

    logger.info("-" * 60)
    logger.info("Precision Verification:")
    logger.info("-" * 60)

    all_pass = True
    for rank in range(world_size):
        logger.info(f"\n[Rank {rank}] 开始精度对比...")

        golden_y, golden_receive_cnt = gen_golden_data(
            rank, x_per_rank, dstrank_id_per_rank, test_max_bs, test_data_size
        )

        npu_y, npu_receive_cnt = npu_outputs[rank]

        valid_rows = int(golden_receive_cnt.sum().item())
        y_pass = compare_result(npu_y[:valid_rows], golden_y[:valid_rows], f"rank_{rank}_y")
        receive_cnt_pass = compare_result(
            npu_receive_cnt, golden_receive_cnt, f"rank_{rank}_receive_cnt"
        )

        if not (y_pass and receive_cnt_pass):
            all_pass = False

    return all_pass


def run_bandwidth_test(
    x_per_rank: List[torch.Tensor],
    dstrank_id_per_rank: List[torch.Tensor],
    test_max_bs: int,
    test_data_size: int,
) -> Dict[str, Any]:
    """运行带宽测试"""
    logger.info("=" * 60)
    logger.info("Bandwidth Test")
    logger.info("=" * 60)

    clean_previous_profiling_result()

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        p = Process(
            target=run_bandwidth_test_npu_bandwidth,
            args=(
                result_queue,
                rank,
                x_per_rank[rank],
                dstrank_id_per_rank[rank],
                test_max_bs,
                mode,
                world_size,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    npu_outputs = {}
    while not result_queue.empty():
        rank, output = result_queue.get()
        if output is not None:
            npu_outputs[rank] = output
        else:
            logger.error(f"[ERROR] Rank {rank} 带宽测试执行失败")
            return None

    data_size_bytes = test_data_size * dtype_bytes
    perf_result = get_bandwidth_performance(
        CONFIG.run.profiling_path, data_size_bytes, world_size, len(x_per_rank[0])
    )

    return perf_result


def shared_jetty_batch_tokens(dstrank_id_per_rank, test_bs, ep_world_size, test_mode,
                              shared_jetty_num, requested_batch_tokens):
    """Largest actual payload batch after the kernel's BS split, excluding local copies."""
    if ep_world_size == 1:
        return 0
    capacity = requested_batch_tokens or URMA_MAX_BATCH_TOKENS
    if test_mode == 3:
        return min(capacity, (test_bs // ep_world_size + shared_jetty_num - 1) // shared_jetty_num)
    largest = 0
    base, remainder = divmod(test_bs, shared_jetty_num)
    for rank, destinations in enumerate(dstrank_id_per_rank):
        start = 0
        for core in range(shared_jetty_num):
            end = start + base + (core < remainder)
            counts = torch.bincount(destinations[start:end].to(torch.int64), minlength=ep_world_size)
            counts[rank] = 0
            largest = max(largest, int(counts.max()))
            start = end
    return min(capacity, largest)


def run_multi_config_bandwidth_test() -> List[Dict[str, Any]]:
    """运行多配置带宽测试（共享通信域）"""
    all_results = []

    clean_previous_profiling_result()

    all_inputs = []
    for i, cfg in enumerate(test_configs):
        test_bs = cfg["bs"]
        test_data_size = cfg["data_size"]
        test_max_bs = cfg["max_bs"]
        x_per_rank, dstrank_id_per_rank = generate_inputs(test_bs, test_data_size)
        all_inputs.append(
            (x_per_rank, dstrank_id_per_rank, test_max_bs, test_data_size, i + 1)
        )

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        rank_configs = [
            (
                all_inputs[i][0][rank],
                all_inputs[i][1][rank],
                all_inputs[i][2],
                all_inputs[i][3],
                world_size,
                all_inputs[i][4],
            )
            for i in range(len(all_inputs))
        ]
        p = Process(
            target=run_bandwidth_test_npu_multi_config,
            args=(result_queue, rank, rank_configs),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    npu_all_results = {}
    while not result_queue.empty():
        rank, results = result_queue.get()
        if results is not None:
            npu_all_results[rank] = results
        else:
            logger.error(f"[ERROR] Rank {rank} 多配置测试执行失败")
            return []

    for i, cfg in enumerate(test_configs):
        test_data_size = cfg["data_size"]
        test_bs = cfg["bs"]
        data_size_bytes = test_data_size * dtype_bytes
        config_index = i + 1

        config_profiling_path = f"{CONFIG.run.profiling_path}_config_{config_index}"
        perf_result = get_bandwidth_performance(
            config_profiling_path, data_size_bytes, world_size, test_bs
        )

        if perf_result:
            perf_result["config_index"] = config_index
            perf_result["config"] = cfg
            if CONFIG.model.shared_jetty:
                perf_result["shared_jetty_batch_tokens"] = shared_jetty_batch_tokens(
                    all_inputs[i][1], test_bs, world_size, mode,
                    CONFIG.model.shared_jetty_num, CONFIG.model.batch_tokens)
            all_results.append(perf_result)

    print_summary_table(all_results)

    return all_results


def print_summary_table(results: List[Dict[str, Any]]):
    """打印汇总结果表格"""
    logger.info("=" * 90)
    logger.info("Bandwidth Test Summary - All Configurations")
    logger.info("=" * 90)

    if not results:
        logger.warning("[WARN] No valid test results")
        return

    logger.info("Communication: %s; mode: %d; requested batch_tokens: %d", CONFIG.model.comm_alg, mode,
                CONFIG.model.batch_tokens)
    header = (
        f"{'No.':<5} {'BS':<8} {'H':<10} {'Rank Data(MB)':<16} "
        f"{'Batch Tokens':<14} {'Batch Data(MB)':<16} "
        f"{'Mean BW(GB/s)':<16} {'Max BW(GB/s)':<16} {'Mean Time(us)':<14}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    max_bandwidth = 0
    max_bw_config = None

    for r in results:
        cfg = r["config"]
        token_bytes = cfg["data_size"] * dtype_bytes
        rank_data_mb = cfg["bs"] * token_bytes / (1024 * 1024)
        if mode == 3:
            rank_data_mb = (cfg["bs"] // world_size) * (world_size - 1) * token_bytes / (1024 * 1024)
        if CONFIG.model.comm_alg != "urma":
            batch_tokens_text = batch_data_mb_text = "N/A"
        elif CONFIG.model.shared_jetty:
            batch_tokens = r["shared_jetty_batch_tokens"]
            batch_tokens_text = str(batch_tokens)
            batch_data_mb_text = f"{batch_tokens * token_bytes / (1024 * 1024):.4f}"
        else:
            batch_capacity = CONFIG.model.batch_tokens or URMA_MAX_BATCH_TOKENS
            max_peer_tokens = cfg["bs"] // world_size
            # Business inputs distribute the remainder; mode 3 sends only floor(BS/world_size).
            if mode != 3 and cfg["bs"] % world_size:
                max_peer_tokens += 1
            batch_tokens = min(batch_capacity, max_peer_tokens) if world_size > 1 else 0
            batch_tokens_text = str(batch_tokens)
            batch_data_mb_text = f"{batch_tokens * token_bytes / (1024 * 1024):.4f}"
        row = (
            f"{r['config_index']:<5} {cfg['bs']:<8} {cfg['data_size']:<10} "
            f"{rank_data_mb:<16.4f} {batch_tokens_text:<14} {batch_data_mb_text:<16} "
            f"{r['mean_bandwidth_gbps']:<16.2f} "
            f"{r['max_bandwidth_gbps']:<16.2f} {r['mean_time_us']:<14.2f}"
        )
        logger.info(row)

        if r["max_bandwidth_gbps"] > max_bandwidth:
            max_bandwidth = r["max_bandwidth_gbps"]
            max_bw_config = r

    logger.info("-" * len(header))
    logger.info("MB = 1024^2 bytes. Mode 0/1 Rank Data includes local destinations.")
    logger.info("Mode 3 Rank Data = floor(BS/world_size) * (world_size-1) * H * dtype bytes; only remote payload.")
    logger.info(
        "Batch Tokens/Data show the largest actual data commit to one remote destination; smaller tails may exist.")
    logger.info("Shared jetty Batch Tokens account for the BS split across shared_jetty_num cores.")
    logger.info("Ordinary URMA mode 3 Batch Tokens = min(selected batch, floor(BS/world_size)).")
    logger.info("URMA batch_tokens=0 uses the current tiling maximum of 2048. N/A: MTE has no URMA batch submission.")

    if max_bw_config:
        logger.info(f"\n{'*' * 90}")
        logger.info(
            f"Maximum Bandwidth: {max_bw_config['max_bandwidth_gbps']:.2f} GB/s"
        )
        logger.info(f"  H: {max_bw_config['config']['data_size']}")
        best_cfg = max_bw_config["config"]
        best_rank_data_mb = best_cfg["bs"] * best_cfg["data_size"] * dtype_bytes / (1024 * 1024)
        if mode == 3:
            best_rank_data_mb = max_bw_config["single_rank_data_kb"] / 1024
        logger.info(f"  Rank Data: {best_rank_data_mb:.4f} MB")
        logger.info(f"  Mean Time: {max_bw_config['mean_time_us']:.2f} us")
        logger.info(f"{'*' * 90}")


# endregion

# region Main


def main():
    print_setting()
    if CONFIG.run.is_check_setting:
        check_setting()

    if CONFIG.run.is_bandwidth_test and len(test_configs) > 1:
        all_results = run_multi_config_bandwidth_test()

        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(
            f"Total configurations tested: {len(all_results)}/{len(test_configs)}"
        )
        logger.info("=" * 60)
    else:
        bs = test_configs[0]["bs"]
        data_size = test_configs[0]["data_size"]
        max_bs = test_configs[0]["max_bs"]
        x_per_rank, dstrank_id_per_rank = generate_inputs(bs, data_size)

        precision_pass = True
        bandwidth_result = None

        if CONFIG.run.is_precision_test:
            precision_pass = run_precision_test(
                x_per_rank, dstrank_id_per_rank, max_bs, data_size
            )

        if CONFIG.run.is_bandwidth_test:
            bandwidth_result = run_bandwidth_test(
                x_per_rank, dstrank_id_per_rank, max_bs, data_size
            )

        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)

        if CONFIG.run.is_precision_test:
            logger.info(f"Precision Test: {'PASS' if precision_pass else 'FAIL'}")

        if CONFIG.run.is_bandwidth_test:
            if bandwidth_result:
                logger.info("Bandwidth Test: Completed")
                logger.info(
                    f"  Mean Bandwidth: {bandwidth_result['mean_bandwidth_gbps']:.2f} GB/s"
                )
                logger.info(
                    f"  Max Bandwidth:  {bandwidth_result['max_bandwidth_gbps']:.2f} GB/s"
                )
            else:
                logger.error("Bandwidth Test: FAILED")

        logger.info("=" * 60)

        if CONFIG.run.is_precision_test and not precision_pass:
            exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
