from __future__ import annotations

import csv
import os
import shutil
import socket
import time
import math
import traceback
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Process

from golden import (
    Feature,
    GoldenHandle,
    feature_main,
    feature_scales,
    gen_combine_golden_cpu,
    gen_dispatch_golden_cpu,
    is_tuple_feature,
)
from compare import (
    CompareConfig,
    compare_combine_outputs as compare_combine_outputs_impl,
    compare_dispatch_outputs as compare_dispatch_outputs_impl,
    print_summary as print_summary_impl,
)
from cache import (
    dtype_from_name,
    load_case_cache as load_case_cache_file,
    save_case_cache as save_case_cache_file,
    tensor_to_storage_numpy,
)


def _is_fp8_dtype_name(dtype_name: str) -> bool:
    return dtype_name in ("fp8_e4m3fn", "float8_e4m3fn", "fp8_e5m2", "float8_e5m2")


def _is_supported_x_dtype_name(dtype_name: str) -> bool:
    return dtype_name in (
        "bfloat16",
        "bf16",
        "float16",
        "fp16",
        "fp8_e4m3fn",
        "float8_e4m3fn",
        "fp8_e5m2",
        "float8_e5m2",
    )


def _is_fp32_scales_dtype_name(dtype_name: str) -> bool:
    return dtype_name in ("float32", "float")


def _is_e8m0_scales_dtype_name(dtype_name: str) -> bool:
    return dtype_name in ("fp8_e8m0", "float8_e8m0", "float8_e8m0fnu")


def _is_supported_scales_dtype_name(dtype_name: str) -> bool:
    return _is_fp32_scales_dtype_name(dtype_name) or _is_e8m0_scales_dtype_name(dtype_name)


def _align_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _default_dim_scales(hidden_size: int, scales_dtype_name: str) -> int:
    if _is_fp32_scales_dtype_name(scales_dtype_name):
        return math.ceil(hidden_size / 128)
    return _align_even(math.ceil(hidden_size / 32))


def _cpu_generator(seed_offset: int = 0) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed + seed_offset)
    return generator

# ======================================================================
# region torch_extension编译缓存复用
# ======================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "torch_extension_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir

from cann_ops_transformer.ops import ElasticBuffer, EPHandle

# endregion

# ======================================================================
# region 控制参数配置
# ======================================================================

# region 执行哪些测试

is_dispatch = 1
is_combine = 0
is_continue = 0

# endregion

# region CPU / 缓存模式

is_cpu_only = False
is_reuse_golden = True
is_force_generate_golden = False
is_save_golden = False
strict_dispatch_metadata_order = False
check_invalid_padding = False
compare_print_detail_max_elements = 16

# endregion

# region 性能检测相关

is_performance = 0
loop = 105
repeat = 1
warmup = 5
performance_tail_count = 50

# endregion

# region 控核配置

is_full_core = True
core_num = 16 if not is_full_core else 32
aic_core_num = core_num
aiv_core_num = 2 * core_num

# endregion

# region 算子核心参数

num_tokens = 16
hidden = 1024
num_topk = 2
num_experts = 4
num_max_tokens_per_rank = num_tokens
expert_alignment = 1
is_variable_bs = False  # 可变 bs
is_topk_balance = False  # 是否让 topK 专家按目标 rank 均匀分布

# bfloat16 / float16 / fp8_e4m3fn / fp8_e5m2
x_dtype = "bfloat16"
# fp8 输入时随机选择 float32/fp8_e8m0 scales dtype，不使能时使用 scales_dtype
is_random_scales_dtype = False
# float32 / fp8_e8m0，仅在 is_random_scales_dtype 不使能时生效
scales_dtype = "float32"
dim_scales = _default_dim_scales(hidden, scales_dtype)
is_x_random = True
random_seed = 1099

dispatch_with_handle = False
dispatch_do_cpu_sync = False
# False 时 dispatch 不传 topk_weights；cache 模式固定不传且不比较
dispatch_with_topk_weights = True
# True 时 combine 独立传 expanded topk weights，不依赖 dispatch 是否传 topk_weights
combine_with_topk_weights = False
is_continue_use_dispatch_output = True

# endregion

# region 通信域配置

server_num = 1
rank_per_dev = 2
is_cloud = False
if not is_cloud:
    server_index = 0 if server_num > 1 else 0
    master_ip = "71.20.45.123" if server_num > 1 else "127.0.0.1"
else:
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

world_size = server_num * rank_per_dev
server_ranks = slice(server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev)

# endregion

# region 测试脚本执行相关

is_save_tensor_to_bin = False
# 是否自动保存精度/性能结果到 csv_result
is_save_result_to_csv = True
is_check_setting = True
is_clear_profiling = True


def dispatch_do_cpu_sync_value() -> bool:
    return bool(dispatch_do_cpu_sync)


def build_current_case_name(id: int) -> str:
    return f"MOE_EP_TRAIN_nt{num_tokens}_vbs{int(is_variable_bs)}_h{hidden}_topk{num_topk}_experts{num_experts}_nmt{num_max_tokens_per_rank}_balance{int(is_topk_balance)}_xdtype{x_dtype}_sdtype{scales_dtype}_rsdtype{int(is_random_scales_dtype)}_dwt{int(dispatch_with_topk_weights)}_sync{int(dispatch_do_cpu_sync_value())}_cached{int(dispatch_with_handle)}_ranks{world_size}_ID{id:04d}"


current_case_name = build_current_case_name(1)

if os.environ.get("MOE_EP_TRAIN_CPU_ONLY") is not None:
    is_cpu_only = bool(int(os.environ["MOE_EP_TRAIN_CPU_ONLY"]))

buff_size = ElasticBuffer.get_moe_ep_ccl_buffer_size(world_size, num_max_tokens_per_rank, hidden, num_experts, num_topk)
if buff_size >= 200:
    os.environ['HCCL_BUFFSIZE'] = f'{buff_size}'

# endregion

# endregion


# ======================================================================
# region 通用工具
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
GOLDEN_CACHE_ROOT = SCRIPT_DIR / "golden_cache"
dispatch_profiling_path = SCRIPT_DIR / "Profiling_Result_train_dispatch"
combine_profiling_path = SCRIPT_DIR / "Profiling_Result_train_combine"
continue_profiling_path = SCRIPT_DIR / "Profiling_Result_train_continue"

dispatch_op_types = ["MoeEpDispatch", "MoeEpDispatchEpilogue"]
dispatch_total_op_type = "MoeEpDispatchTotal"
combine_op_types = ["MoeEpCombine"]
continue_op_types = dispatch_op_types + combine_op_types
profiling_file_time_window_seconds = 30
COMBINE_TIMING_SLOTS = (
    "send_token_total",
    "set_status_total",
    "token_write_nbi",
    "status_drain_before",
    "status_write_nbi",
    "status_drain_after",
)
COMBINE_FLOW_STAGES = (
    "token_prepare_and_copy",
    "token_write_nbi",
    "status_prepare_and_copy",
    "status_drain_before",
    "status_write_nbi",
    "status_drain_after",
)
COMBINE_TIMING_CYCLES_PER_US = 1000  # Ascend 950


def extract_combine_kernel_timing(
    output_tensor: torch.Tensor,
    num_cores: int,
    rank: int,
    out_dir: Optional[Path] = None,
    cycles_per_us: int = COMBINE_TIMING_CYCLES_PER_US,
) -> torch.Tensor:
    """Export cumulative MoeEpCombine timing stored at the start of combined_x."""
    timing_slots = 16
    timing_bytes = num_cores * timing_slots * 2 * 8
    available_bytes = output_tensor.numel() * output_tensor.element_size()
    if available_bytes < timing_bytes:
        raise ValueError(
            f"combined_x has {available_bytes} bytes, but kernel timing needs {timing_bytes} bytes"
        )

    if out_dir is None:
        out_dir = SCRIPT_DIR / "kernel_timing_output"
    rank_dir = Path(out_dir) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    # The kernel writes uint64_t values. Reinterpreting them as int64 would turn
    # values with bit 63 set into misleading negative timestamps/durations.
    timing = (
        output_tensor.contiguous()
        .view(torch.uint8)
        .reshape(-1)[:timing_bytes]
        .cpu()
        .view(torch.uint64)
        .reshape(num_cores, timing_slots, 2)
    )

    raw_path = rank_dir / f"combine_timing_rank_{rank}_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["core", "slot", "name", "last_start_cycle", "accumulated_cycles", "accumulated_us"]
        )
        for core in range(num_cores):
            for slot, name in enumerate(COMBINE_TIMING_SLOTS):
                accumulated_cycles = int(timing[core, slot, 1].item())
                writer.writerow(
                    [
                        core,
                        slot,
                        name,
                        int(timing[core, slot, 0].item()),
                        accumulated_cycles,
                        round(accumulated_cycles / cycles_per_us, 3),
                    ]
                )

    duration_path = rank_dir / f"combine_timing_rank_{rank}_duration.csv"
    with duration_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["core", *COMBINE_TIMING_SLOTS])
        for core in range(num_cores):
            writer.writerow(
                [core]
                + [
                    round(int(timing[core, slot, 1].item()) / cycles_per_us, 3)
                    for slot in range(len(COMBINE_TIMING_SLOTS))
                ]
            )

    # Slots 0/1 are inclusive totals and contain their communication sub-slots.
    # Subtract those nested timings to build mutually exclusive accumulated
    # components. This is a logical flow breakdown, not a per-token event trace.
    flow_rows = []
    flow_durations_by_core = []
    for core in range(num_cores):
        send_token_cycles = int(timing[core, 0, 1].item())
        set_status_cycles = int(timing[core, 1, 1].item())
        token_write_cycles = int(timing[core, 2, 1].item())
        drain_before_cycles = int(timing[core, 3, 1].item())
        status_write_cycles = int(timing[core, 4, 1].item())
        drain_after_cycles = int(timing[core, 5, 1].item())
        token_prepare_cycles = max(send_token_cycles - token_write_cycles, 0)
        status_prepare_cycles = max(
            set_status_cycles - drain_before_cycles - status_write_cycles - drain_after_cycles,
            0,
        )
        stage_cycles = (
            token_prepare_cycles,
            token_write_cycles,
            status_prepare_cycles,
            drain_before_cycles,
            status_write_cycles,
            drain_after_cycles,
        )
        flow_durations_by_core.append(
            [stage_cycle / cycles_per_us for stage_cycle in stage_cycles]
        )
        total_cycles = sum(stage_cycles)
        logical_offset_cycles = 0
        for stage_name, stage_cycle in zip(COMBINE_FLOW_STAGES, stage_cycles):
            flow_rows.append(
                {
                    "core": core,
                    "stage": stage_name,
                    "logical_start_us": logical_offset_cycles / cycles_per_us,
                    "duration_us": stage_cycle / cycles_per_us,
                    "share_percent": 100.0 * stage_cycle / total_cycles if total_cycles else 0.0,
                }
            )
            logical_offset_cycles += stage_cycle

    flow_path = rank_dir / f"combine_timing_rank_{rank}_send_flow.csv"
    with flow_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=flow_rows[0].keys())
        writer.writeheader()
        writer.writerows(flow_rows)

    flow_summary_rows = []
    for stage_index, stage_name in enumerate(COMBINE_FLOW_STAGES):
        stage_durations = [durations[stage_index] for durations in flow_durations_by_core]
        slowest_core = max(range(num_cores), key=lambda core: stage_durations[core])
        flow_summary_rows.append(
            {
                "stage": stage_name,
                "avg_us": sum(stage_durations) / len(stage_durations),
                "max_us": stage_durations[slowest_core],
                "slowest_core": slowest_core,
            }
        )
    flow_summary_path = rank_dir / f"combine_timing_rank_{rank}_send_flow_summary.csv"
    with flow_summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=flow_summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(flow_summary_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        inclusive_durations_us = [
            [
                int(timing[core, slot, 1].item()) / cycles_per_us
                for core in range(num_cores)
            ]
            for slot in range(len(COMBINE_TIMING_SLOTS))
        ]
        fig, ax = plt.subplots(figsize=(14, max(5, num_cores * 0.3)))
        bar_height = 0.12
        core_positions = list(range(num_cores))
        for slot, name in enumerate(COMBINE_TIMING_SLOTS):
            offset = (slot - (len(COMBINE_TIMING_SLOTS) - 1) / 2) * bar_height
            ax.barh(
                [core + offset for core in core_positions],
                inclusive_durations_us[slot],
                height=bar_height,
                label=name,
            )
        ax.set_yticks(core_positions)
        ax.set_yticklabels([f"core_{core}" for core in range(num_cores)], fontsize=7)
        ax.set_xlabel("accumulated time (us)")
        ax.set_title(f"MoeEpCombine cumulative kernel timing - rank {rank}")
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(rank_dir / f"combine_timing_rank_{rank}_bars.png", dpi=150)
        plt.close(fig)

        fig, (summary_ax, flow_ax) = plt.subplots(
            2,
            1,
            figsize=(16, max(10, num_cores * 0.3 + 5)),
            gridspec_kw={"height_ratios": [1, max(2, num_cores * 0.08)]},
        )
        stage_positions = list(range(len(COMBINE_FLOW_STAGES)))
        summary_width = 0.35
        summary_ax.bar(
            [position - summary_width / 2 for position in stage_positions],
            [row["avg_us"] for row in flow_summary_rows],
            width=summary_width,
            label="average",
        )
        summary_ax.bar(
            [position + summary_width / 2 for position in stage_positions],
            [row["max_us"] for row in flow_summary_rows],
            width=summary_width,
            label="maximum",
        )
        summary_ax.set_xticks(stage_positions)
        summary_ax.set_xticklabels(COMBINE_FLOW_STAGES, rotation=15, ha="right")
        summary_ax.set_ylabel("duration (us)")
        summary_ax.set_title(f"MoeEpCombine accumulated send-flow summary - rank {rank}")
        summary_ax.legend()
        summary_ax.grid(axis="y", alpha=0.25)

        colors = plt.cm.Set2.colors
        left = [0.0] * num_cores
        for stage_index, stage_name in enumerate(COMBINE_FLOW_STAGES):
            stage_durations = [durations[stage_index] for durations in flow_durations_by_core]
            flow_ax.barh(
                core_positions,
                stage_durations,
                left=left,
                height=0.72,
                color=colors[stage_index % len(colors)],
                label=stage_name,
            )
            left = [offset + duration for offset, duration in zip(left, stage_durations)]
        flow_ax.set_yticks(core_positions)
        flow_ax.set_yticklabels([f"core_{core}" for core in core_positions], fontsize=7)
        flow_ax.set_xlabel("accumulated logical time (us)")
        flow_ax.set_ylabel("AIV core")
        flow_ax.set_title(
            "Per-core accumulated logical breakdown (exclusive components, not wall-clock timeline)"
        )
        flow_ax.invert_yaxis()
        flow_ax.legend(fontsize=7, ncol=3)
        flow_ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(rank_dir / f"combine_timing_rank_{rank}_send_flow.png", dpi=150)
        plt.close(fig)
    except Exception as error:
        print(f"[TIMING] rank {rank}: skipped timing chart: {error}")

    flow_hotspot = max(flow_summary_rows, key=lambda row: row["max_us"])
    print(
        f"[TIMING] rank {rank}: send-flow hotspot={flow_hotspot['stage']}, "
        f"max={flow_hotspot['max_us']:.3f} us on core {flow_hotspot['slowest_core']}"
    )
    print(f"[TIMING] rank {rank}: combine timing saved to {rank_dir}")
    return timing


def write_row_to_csv(csv_name: str, row_content: list) -> None:
    csv_dir = REPO_ROOT / "csv_result"
    csv_dir.mkdir(exist_ok=True)
    with open(csv_dir / csv_name, "a", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(row_content)


def write_csv_header_once(csv_name: str, title: list) -> None:
    csv_path = REPO_ROOT / "csv_result" / csv_name
    if not csv_path.exists():
        write_row_to_csv(csv_name, title)


def _csv_record_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_accuracy_result(tag: str, result: Dict[str, bool]) -> None:
    if not is_save_result_to_csv:
        return
    rows = getattr(result, "csv_rows", [])
    if not rows:
        res_output = ("FAIL", 0.0, f"{tag} 未生成 toolss 对比结果")
        write_row_to_csv("moe_distribute.csv", [current_case_name, f"{tag}.toolss_result_missing", res_output])
        raise RuntimeError(f"{tag} 未生成 toolss 对比结果")
    for tensor_name, res_output in rows:
        write_row_to_csv("moe_distribute.csv", [current_case_name, tensor_name, res_output])


def save_performance_result(tag: str, performance: Dict[str, Dict[str, Any]], op_types: List[str]) -> None:
    if not is_save_result_to_csv or not performance:
        return
    recordtime = _csv_record_time()
    csv_name = f"{tag}_performance_result.csv"
    title = [
        "op_type",
        "case_name",
        "mc2_shared_average",
        "mc2_moe_average",
        "mc2_shared_min",
        "mc2_moe_min",
        "recordtime",
    ]
    for op_type, item in _ordered_performance_items(performance, op_types):
        write_csv_header_once(csv_name, title)
        write_row_to_csv(
            csv_name,
            [
                op_type,
                current_case_name,
                item["mean_us"],
                item["mean_us"],
                item["min_us"],
                item["min_us"],
                recordtime,
            ],
        )


def resolve_scales_dtype_cpu() -> str:
    if not (_is_fp8_dtype_name(x_dtype) and is_random_scales_dtype):
        return scales_dtype
    index = int(torch.randint(0, 2, (1,), generator=_cpu_generator(5000), dtype=torch.int64).item())
    return ("float32", "fp8_e8m0")[index]


def resolve_dim_scales_cpu(effective_scales_dtype: Optional[str] = None) -> int:
    if _is_fp8_dtype_name(x_dtype) and is_random_scales_dtype:
        effective_scales_dtype = effective_scales_dtype or resolve_scales_dtype_cpu()
        return _default_dim_scales(hidden, effective_scales_dtype)
    return dim_scales


def golden_script_paths() -> List[Path]:
    return [
        Path(__file__).resolve(),
        SCRIPT_DIR / "golden.py",
        SCRIPT_DIR / "cache.py",
        SCRIPT_DIR / "compare.py",
        SCRIPT_DIR / "toolss.py",
    ]


def case_params() -> Dict[str, Any]:
    return {
        "num_tokens": num_tokens,
        "is_variable_bs": is_variable_bs,
        "hidden": hidden,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "expert_alignment": expert_alignment,
        "is_topk_balance": is_topk_balance,
        "x_dtype": x_dtype,
        "scales_dtype": scales_dtype,
        "effective_scales_dtype": resolve_scales_dtype_cpu(),
        "is_random_scales_dtype": is_random_scales_dtype,
        "dim_scales": dim_scales,
        "effective_dim_scales": resolve_dim_scales_cpu(),
        "random_seed": random_seed,
        "world_size": world_size,
        "dispatch_with_topk_weights": dispatch_with_topk_weights,
        "dispatch_do_cpu_sync": dispatch_do_cpu_sync_value(),
        "combine_with_topk_weights": combine_with_topk_weights,
    }


def check_setting() -> None:
    def err(msg: str) -> None:
        raise ValueError(msg)

    if world_size <= 0:
        err("world_size 必须大于 0")
    if num_experts % world_size != 0:
        err("num_experts 必须能整除 world_size")
    if num_topk <= 0:
        err("num_topk 必须大于 0")
    if num_experts < num_topk:
        err("num_experts 不能小于 num_topk")
    if num_tokens <= 0:
        err("num_tokens 必须大于 0")
    if num_tokens > num_max_tokens_per_rank:
        err("num_tokens 不能超过 num_max_tokens_per_rank")
    if expert_alignment != 1:
        err("当前训练接口 expert_alignment 必须为 1")
    if not _is_supported_x_dtype_name(x_dtype):
        err(f"不支持的 x_dtype: {x_dtype}")
    effective_scales_dtype = resolve_scales_dtype_cpu()
    effective_dim_scales = resolve_dim_scales_cpu(effective_scales_dtype)
    if _is_fp8_dtype_name(x_dtype) and effective_dim_scales <= 0:
        err("fp8 tuple 输入要求 dim_scales > 0")
    if _is_fp8_dtype_name(x_dtype) and not _is_supported_scales_dtype_name(effective_scales_dtype):
        err(f"不支持的 scales_dtype: {effective_scales_dtype}")
    if dispatch_with_handle and dispatch_do_cpu_sync_value():
        print("[WARNING] cached dispatch 会强制 do_cpu_sync=False")
    if is_performance:
        if repeat <= 0:
            err("repeat 必须大于 0")
        active = loop // repeat - warmup
        if active <= 0:
            err("性能模式要求 loop // repeat > warmup")
        if performance_tail_count < 0:
            err("performance_tail_count 不能小于 0")
    if core_num <= 0:
        err("core_num 必须大于 0")
    if aic_core_num != core_num:
        err("aic_core_num 必须等于 core_num")
    if aiv_core_num != 2 * core_num:
        err("aiv_core_num 必须等于 2 * core_num")


def print_setting() -> None:
    print("==================== Test Setting ====================")
    print(f"is_dispatch={is_dispatch}, is_combine={is_combine}, is_continue={is_continue}")
    print(
        f"is_performance={is_performance}, loop={loop}, repeat={repeat}, "
        f"warmup={warmup}, performance_tail_count={performance_tail_count}"
    )
    print(f"is_full_core={is_full_core}, core_num={core_num}, aic_core_num={aic_core_num}, aiv_core_num={aiv_core_num}")
    print(f"num_tokens={num_tokens}, is_variable_bs={is_variable_bs}, hidden={hidden}, num_topk={num_topk}")
    print(
        f"num_experts={num_experts}, num_max_tokens_per_rank={num_max_tokens_per_rank}, "
        f"is_topk_balance={is_topk_balance}"
    )
    print(
        f"x_dtype={x_dtype}, scales_dtype={scales_dtype}, "
        f"is_random_scales_dtype={is_random_scales_dtype}, "
        f"effective_scales_dtype={resolve_scales_dtype_cpu()}, "
        f"dim_scales={dim_scales}, effective_dim_scales={resolve_dim_scales_cpu()}"
    )
    print(f"world_size={world_size}, rank_per_dev={rank_per_dev}, server_index={server_index}")
    print(
        f"dispatch_with_handle={dispatch_with_handle}, "
        f"dispatch_with_topk_weights={dispatch_with_topk_weights}, "
        f"dispatch_do_cpu_sync={dispatch_do_cpu_sync_value()}"
    )
    print(f"combine_with_topk_weights={combine_with_topk_weights}")
    print(f"compare_print_detail_max_elements={compare_print_detail_max_elements}")
    print(f"is_save_result_to_csv={is_save_result_to_csv}")
    print(f"is_reuse_golden={is_reuse_golden}, is_force_generate_golden={is_force_generate_golden}, is_save_golden={is_save_golden}")
    print(f"current_case_name={current_case_name}")


def save_tensor_to_bin(tensor: torch.Tensor, bin_name: str) -> None:
    data_dir = SCRIPT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    tensor_to_storage_numpy(tensor).tofile(data_dir / f"{bin_name}.bin")


def clean_profiling_result(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def clean_previous_profiling_result(paths: List[Path]) -> None:
    if not is_performance or not is_clear_profiling:
        return
    for path in paths:
        clean_profiling_result(path)


def get_latest_profiling_file_infos(profiling_path: Path, file_prefix: str) -> List[Path]:
    file_infos = []
    if not profiling_path.exists():
        print(f"[WARNING] profiling 目录不存在: {profiling_path}")
        return []
    for path in profiling_path.rglob(f"{file_prefix}*"):
        if not path.is_file():
            continue
        try:
            file_infos.append((path.stat().st_mtime, path))
        except OSError as err:
            print(f"[WARNING] 获取 profiling 文件时间失败，已跳过: {path}, {err}")
    if not file_infos:
        print(f"[WARNING] 未找到 {profiling_path} 下的 {file_prefix}* 文件")
        return []
    file_infos.sort(key=lambda item: item[0], reverse=True)
    latest_mtime = file_infos[0][0]
    return [
        path
        for mtime, path in file_infos
        if latest_mtime - mtime <= profiling_file_time_window_seconds
    ]


def _get_first_existing(row: Dict[str, str], names: List[str]) -> Optional[str]:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _select_performance_values(values: List[float]) -> List[float]:
    return values[-performance_tail_count:] if performance_tail_count > 0 else values


def _summarize_performance_values(values: List[float], total_count: int) -> Dict[str, Any]:
    return {
        "mean_us": sum(values) / len(values),
        "min_us": min(values),
        "max_us": max(values),
        "count": float(len(values)),
        "total_count": float(total_count),
        "samples": values,
    }


def _add_dispatch_total_performance(performance: Dict[str, Dict[str, Any]]) -> None:
    if not all(op_type in performance for op_type in dispatch_op_types):
        return
    sample_lists = [performance[op_type]["samples"] for op_type in dispatch_op_types]
    sample_count = min(len(samples) for samples in sample_lists)
    if sample_count <= 0:
        return
    paired_samples = [samples[-sample_count:] for samples in sample_lists]
    total_samples = [sum(samples[index] for samples in paired_samples) for index in range(sample_count)]
    total_count = int(min(performance[op_type]["total_count"] for op_type in dispatch_op_types))
    performance[dispatch_total_op_type] = _summarize_performance_values(total_samples, total_count)


def _ordered_performance_items(
    performance: Dict[str, Dict[str, Any]],
    op_types: List[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    ordered_names = []
    for op_type in op_types:
        ordered_names.append(op_type)
        if op_type == "MoeEpDispatchEpilogue" and dispatch_total_op_type in performance:
            ordered_names.append(dispatch_total_op_type)
    ordered_names.extend(name for name in performance if name not in ordered_names)
    return [(name, performance[name]) for name in ordered_names if name in performance]


def _read_op_summary_rows(summary_path: Path) -> List[Dict[str, str]]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_summary_device_id(rows: List[Dict[str, str]]) -> Optional[int]:
    if not rows:
        return None
    device_id_text = _get_first_existing(rows[0], ["Device_id", "Device ID", "DeviceId"])
    if device_id_text is None:
        return None
    try:
        return int(device_id_text)
    except ValueError:
        return None


def get_performance_from_profiling(profiling_path: Path, op_types: List[str]) -> Dict[str, Dict[str, Any]]:
    selected_durations: Dict[str, List[float]] = {op_type: [] for op_type in op_types}
    total_counts: Dict[str, int] = {op_type: 0 for op_type in op_types}
    selected_device_ids = set()
    match_order = sorted(op_types, key=len, reverse=True)
    for summary_path in get_latest_profiling_file_infos(profiling_path, "op_summary"):
        try:
            rows = _read_op_summary_rows(summary_path)
            device_id = _get_summary_device_id(rows)
            if device_id is not None:
                if device_id in selected_device_ids:
                    continue
                if len(selected_device_ids) >= rank_per_dev:
                    break
                selected_device_ids.add(device_id)

            file_durations: Dict[str, List[float]] = {op_type: [] for op_type in op_types}
            for row in rows:
                op_name = _get_first_existing(row, ["OP Type", "Op Type", "Name", "Op Name"])
                duration_text = _get_first_existing(
                    row,
                    ["Task Duration(us)", "Task Duration(us) ", "Duration(us)", "Task Duration"],
                )
                if op_name is None or duration_text is None:
                    continue
                for op_type in match_order:
                    if str(op_name).startswith(op_type):
                        try:
                            file_durations[op_type].append(float(duration_text))
                        except ValueError:
                            pass
                        break
            for op_type, values in file_durations.items():
                total_counts[op_type] += len(values)
                selected_durations[op_type].extend(_select_performance_values(values))
        except Exception as err:
            print(f"[WARNING] 读取 profiling 文件失败，已跳过: {summary_path}, {err}")

    performance: Dict[str, Dict[str, Any]] = {}
    for op_type, values in selected_durations.items():
        if not values:
            continue
        performance[op_type] = _summarize_performance_values(values, total_counts[op_type])
    _add_dispatch_total_performance(performance)
    return performance


def print_performance_result(tag: str, profiling_path: Path, op_types: List[str]) -> Dict[str, Dict[str, Any]]:
    performance = get_performance_from_profiling(profiling_path, op_types)
    print(f"\n################## {tag} performance ##################")
    if not performance:
        print(f"[WARNING] 未解析到 {tag} profiling 性能数据")
        return performance
    for op_type, item in _ordered_performance_items(performance, op_types):
        print(
            f"[INFO] {op_type}: mean={item['mean_us']:.3f} us, "
            f"min={item['min_us']:.3f} us, max={item['max_us']:.3f} us, "
            f"count={int(item['count'])}, total_count={int(item['total_count'])}"
        )
    save_performance_result(tag, performance, op_types)
    return performance


# endregion


# ======================================================================
# region 输入与 golden 生成
# ======================================================================

def gen_real_num_tokens_per_rank_cpu() -> List[int]:
    if not is_variable_bs:
        return [num_tokens for _ in range(world_size)]

    generator = _cpu_generator(4000)
    real_num_tokens = torch.randint(1, num_tokens + 1, (world_size,), generator=generator, dtype=torch.int64).tolist()
    full_rank = int(torch.randint(0, world_size, (1,), generator=generator, dtype=torch.int64).item())
    real_num_tokens[full_rank] = num_tokens
    return [int(item) for item in real_num_tokens]


def _check_real_num_tokens_per_rank(real_num_tokens_per_rank: Sequence[int]) -> None:
    if len(real_num_tokens_per_rank) != world_size:
        raise ValueError("real_num_tokens_per_rank 长度必须等于 world_size")
    for real_num_tokens in real_num_tokens_per_rank:
        if real_num_tokens <= 0:
            raise ValueError("real_num_tokens 必须大于 0")
        if real_num_tokens > num_tokens:
            raise ValueError("real_num_tokens 不能超过 num_tokens")


def _slice_feature_rows(feature: Feature, rows: int) -> Feature:
    if is_tuple_feature(feature):
        return feature_main(feature)[:rows], feature_scales(feature)[:rows]
    return feature[:rows]


def _gen_base_x_cpu(rank: int) -> torch.Tensor:
    generator = _cpu_generator(rank)
    if is_x_random:
        return torch.empty((num_tokens, hidden), dtype=torch.bfloat16).uniform_(-5, 5, generator=generator)
    return torch.arange(num_tokens, dtype=torch.bfloat16).view(-1, 1).repeat(1, hidden)


def _gen_fp8_scales_cpu(rank: int, effective_scales_dtype: str, effective_dim_scales: int) -> torch.Tensor:
    generator = _cpu_generator(3000 + rank)
    if _is_e8m0_scales_dtype_name(effective_scales_dtype):
        # torch 当前没有 e8m0 dtype，按参考脚本使用 uint8 保存 e8m0 编码值。
        if is_x_random:
            return torch.randint(1, 8, (num_tokens, effective_dim_scales), generator=generator, dtype=torch.uint8)
        return torch.ones((num_tokens, effective_dim_scales), dtype=torch.uint8)
    return torch.empty((num_tokens, effective_dim_scales), dtype=torch.float32).uniform_(0.1, 2.0, generator=generator)


def _make_x_feature_cpu(
    rank: int,
    real_num_tokens: int,
    effective_scales_dtype: str,
    effective_dim_scales: int,
) -> Feature:
    base_x = _gen_base_x_cpu(rank)
    if _is_fp8_dtype_name(x_dtype):
        return _slice_feature_rows(
            (
                base_x.to(dtype_from_name(x_dtype)),
                _gen_fp8_scales_cpu(rank, effective_scales_dtype, effective_dim_scales),
            ),
            real_num_tokens,
        )
    return _slice_feature_rows(base_x.to(dtype_from_name(x_dtype)), real_num_tokens)


def gen_x_per_rank_cpu(real_num_tokens_per_rank: Optional[Sequence[int]] = None) -> List[Feature]:
    if not _is_supported_x_dtype_name(x_dtype):
        raise ValueError(f"不支持的 x_dtype: {x_dtype}")
    effective_scales_dtype = resolve_scales_dtype_cpu()
    if _is_fp8_dtype_name(x_dtype) and not _is_supported_scales_dtype_name(effective_scales_dtype):
        raise ValueError(f"不支持的 scales_dtype: {effective_scales_dtype}")
    effective_dim_scales = resolve_dim_scales_cpu(effective_scales_dtype)
    if real_num_tokens_per_rank is None:
        real_num_tokens_per_rank = gen_real_num_tokens_per_rank_cpu()
    _check_real_num_tokens_per_rank(real_num_tokens_per_rank)
    return [
        _make_x_feature_cpu(
            rank,
            int(real_num_tokens_per_rank[rank]),
            effective_scales_dtype,
            effective_dim_scales,
        )
        for rank in range(world_size)
    ]


def gen_topk_idx_per_rank_cpu(real_num_tokens_per_rank: Optional[Sequence[int]] = None) -> List[torch.Tensor]:
    if real_num_tokens_per_rank is None:
        real_num_tokens_per_rank = gen_real_num_tokens_per_rank_cpu()
    _check_real_num_tokens_per_rank(real_num_tokens_per_rank)

    if is_topk_balance:
        # 先按本地专家序号、再按目标 rank 交错排列专家。
        # 例如 world_size=2、num_experts=4 时，顺序为 [0, 2, 1, 3]，
        # 相邻 topK 选择会尽量落到不同目标 rank。
        num_local_experts = num_experts // world_size
        base_order = (
            torch.arange(num_experts, dtype=torch.int32)
            .view(world_size, num_local_experts)
            .transpose(0, 1)
            .reshape(-1)
        )
        topk_idx_per_rank = []
        for rank in range(world_size):
            start = rank * num_tokens * num_topk
            expert_indices = (torch.arange(num_tokens * num_topk, dtype=torch.long) + start) % num_experts
            topk_idx = base_order[expert_indices].view(num_tokens, num_topk)
            topk_idx_per_rank.append(topk_idx[:int(real_num_tokens_per_rank[rank])])
        return topk_idx_per_rank

    topk_idx_per_rank = []
    for rank in range(world_size):
        generator = _cpu_generator(1000 + rank)
        rows = []
        for _ in range(num_tokens):
            rows.append(torch.randperm(num_experts, generator=generator, dtype=torch.int32)[:num_topk])
        topk_idx_per_rank.append(torch.stack(rows, dim=0).to(torch.int32)[:int(real_num_tokens_per_rank[rank])])
    return topk_idx_per_rank


def gen_topk_weights_per_rank_cpu(real_num_tokens_per_rank: Optional[Sequence[int]] = None) -> List[torch.Tensor]:
    if real_num_tokens_per_rank is None:
        real_num_tokens_per_rank = gen_real_num_tokens_per_rank_cpu()
    _check_real_num_tokens_per_rank(real_num_tokens_per_rank)
    topk_weights_per_rank = []
    for rank in range(world_size):
        topk_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32).uniform_(
            0.0,
            1.0,
            generator=_cpu_generator(2000 + rank),
        )
        topk_weights_per_rank.append(topk_weights[:int(real_num_tokens_per_rank[rank])])
    return topk_weights_per_rank


def dispatch_uses_topk_weights() -> bool:
    return not dispatch_with_handle and dispatch_with_topk_weights


def dispatch_topk_weights_for_golden(case_data: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
    if not dispatch_uses_topk_weights():
        return None
    return case_data["topk_weights_per_rank"]


def dispatch_topk_weights_for_npu(case_data: Dict[str, Any]) -> List[Optional[torch.Tensor]]:
    weights = dispatch_topk_weights_for_golden(case_data)
    if weights is None:
        return [None for _ in range(world_size)]
    return weights


def gen_combine_topk_weight_cpu(rank: int, num_expanded_tokens: int) -> torch.Tensor:
    return torch.empty((int(num_expanded_tokens),), dtype=torch.float32).uniform_(
        0.0,
        1.0,
        generator=_cpu_generator(7000 + rank),
    )


def gen_combine_topk_weights_per_rank_cpu(dispatch_goldens: Sequence[Any]) -> List[torch.Tensor]:
    return [
        gen_combine_topk_weight_cpu(rank, feature_main(dispatch_golden.recv_x).shape[0])
        for rank, dispatch_golden in enumerate(dispatch_goldens)
    ]


def combine_topk_weights_for_golden(dispatch_goldens: Sequence[Any]) -> Optional[List[torch.Tensor]]:
    if not combine_with_topk_weights:
        return None
    return gen_combine_topk_weights_per_rank_cpu(dispatch_goldens)


def combine_topk_weights_for_npu(dispatch_goldens: Sequence[Any]) -> List[Optional[torch.Tensor]]:
    weights = combine_topk_weights_for_golden(dispatch_goldens)
    if weights is None:
        return [None for _ in range(world_size)]
    return weights


def generate_case_cpu() -> Dict[str, Any]:
    cached = load_case_cache_file(
        GOLDEN_CACHE_ROOT,
        current_case_name,
        case_params(),
        golden_script_paths(),
        enabled=is_reuse_golden,
        force_generate=is_force_generate_golden,
    )
    if cached is not None:
        print("[INFO] 复用 golden cache")
        return cached

    real_num_tokens_per_rank = gen_real_num_tokens_per_rank_cpu()
    if is_variable_bs:
        print(f"[INFO] variable bs real_num_tokens_per_rank={real_num_tokens_per_rank}")
    x_per_rank = gen_x_per_rank_cpu(real_num_tokens_per_rank)
    topk_idx_per_rank = gen_topk_idx_per_rank_cpu(real_num_tokens_per_rank)
    topk_weights_per_rank = gen_topk_weights_per_rank_cpu(real_num_tokens_per_rank)
    case_inputs = {
        "x_per_rank": x_per_rank,
        "topk_idx_per_rank": topk_idx_per_rank,
        "topk_weights_per_rank": topk_weights_per_rank,
    }
    dispatch_goldens = gen_dispatch_golden_cpu(
        x_per_rank,
        topk_idx_per_rank,
        dispatch_topk_weights_for_golden(case_inputs),
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        do_cpu_sync=False if dispatch_with_handle else dispatch_do_cpu_sync_value(),
    )
    combine_x_per_rank = combine_x_for_dispatch_goldens(dispatch_goldens)
    combine_topk_weights = combine_topk_weights_for_golden(dispatch_goldens)
    combine_goldens = gen_combine_golden_cpu(
        combine_x_per_rank,
        [d.handle for d in dispatch_goldens],
        topk_weights_per_rank=combine_topk_weights,
    )
    case_data = {
        **case_inputs,
        "dispatch_goldens": dispatch_goldens,
        "combine_goldens": combine_goldens,
    }
    if is_save_golden:
        save_case_cache_file(
            GOLDEN_CACHE_ROOT,
            current_case_name,
            case_data,
            case_params(),
            golden_script_paths(),
        )
    return case_data


# endregion


# ======================================================================
# region NPU 调用辅助
# ======================================================================

def set_device(rank: int) -> None:
    import torch_npu

    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank: int):
    print(f"[INFO] device_{rank} 创建 HCCL 通信链路")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50001",
    )
    print(f"device_{rank} init_process_group success")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    if not ep_hcomm_info:
        raise RuntimeError("EP HCCL comm name 为空，请检查 HCCL group 初始化状态")
    return ep_group


def _create_buffer(ep_group):

    return ElasticBuffer(
        ep_group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
    )


def _feature_to_npu(feature: Feature) -> Feature:
    if is_tuple_feature(feature):
        return feature_main(feature).npu(), feature_scales(feature).npu()
    return feature.npu()


def _feature_to_cpu(feature: Feature) -> Feature:
    if is_tuple_feature(feature):
        return feature_main(feature).detach().cpu(), feature_scales(feature).detach().cpu()
    return feature.detach().cpu()


def _split_recv_x_and_scales(recv_x: Feature) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if is_tuple_feature(recv_x):
        return feature_main(recv_x), feature_scales(recv_x)
    return recv_x, None


def _combine_input_dtype() -> torch.dtype:
    dtype_index = int(torch.randint(0, 2, (1,), generator=_cpu_generator()).item())
    return torch.float16 if dtype_index == 0 else torch.bfloat16


def _make_combine_x(recv_x: Feature) -> torch.Tensor:
    recv_x_main, _ = _split_recv_x_and_scales(recv_x)
    if recv_x_main.dtype in (torch.float16, torch.bfloat16):
        return recv_x_main
    return recv_x_main.to(_combine_input_dtype())


def combine_x_for_dispatch_goldens(dispatch_goldens: Sequence[Any]) -> List[torch.Tensor]:
    return [_make_combine_x(dispatch_golden.recv_x) for dispatch_golden in dispatch_goldens]


def _actual_handle_to_golden(handle: Any) -> GoldenHandle:
    num_recv_payloads = getattr(handle, "num_recv_payloads_per_rank", None)
    if num_recv_payloads is None:
        num_recv_payloads = getattr(handle, "num_recv_tokens_per_rank")
    num_expanded_tokens = getattr(handle, "num_expanded_tokens", int(handle.recv_src_metadata.shape[0]))
    return GoldenHandle(
        dst_buffer_slot_idx=handle.dst_buffer_slot_idx.detach().cpu(),
        recv_src_metadata=handle.recv_src_metadata.detach().cpu(),
        num_recv_payloads_per_rank=num_recv_payloads.detach().cpu(),
        num_recv_tokens_per_expert=handle.num_recv_tokens_per_expert.detach().cpu(),
        topk_idx=handle.topk_idx.detach().cpu(),
        num_experts=int(handle.num_experts),
        expert_alignment=int(handle.expert_alignment),
        num_max_tokens_per_rank=int(handle.num_max_tokens_per_rank),
        num_expanded_tokens=int(num_expanded_tokens),
    )


def _golden_handle_to_ep_handle(handle: GoldenHandle):

    field_names = getattr(EPHandle, "__dataclass_fields__", {}).keys()
    kwargs = {}
    device_tensors = {
        "dst_buffer_slot_idx": handle.dst_buffer_slot_idx.npu(),
        "recv_src_metadata": handle.recv_src_metadata.npu(),
        "num_recv_payloads_per_rank": handle.num_recv_payloads_per_rank.npu(),
        "num_recv_tokens_per_rank": handle.num_recv_payloads_per_rank.npu(),
        "num_recv_tokens_per_expert": handle.num_recv_tokens_per_expert.npu(),
        "topk_idx": handle.topk_idx.npu(),
    }
    scalars = {
        "num_experts": handle.num_experts,
        "expert_alignment": handle.expert_alignment,
        "num_max_tokens_per_rank": handle.num_max_tokens_per_rank,
        "num_expanded_tokens": handle.num_expanded_tokens,
    }
    if field_names:
        for name in field_names:
            if name in device_tensors:
                kwargs[name] = device_tensors[name]
            elif name in scalars:
                kwargs[name] = scalars[name]
        return EPHandle(**kwargs)

    return EPHandle(
        device_tensors["dst_buffer_slot_idx"],
        device_tensors["recv_src_metadata"],
        device_tensors["num_recv_tokens_per_rank"],
        device_tensors["num_recv_tokens_per_expert"],
        device_tensors["topk_idx"],
        handle.num_experts,
        handle.expert_alignment,
        handle.num_max_tokens_per_rank,
    )


def get_profiling(path: str):
    if not is_performance:
        return nullcontext()
    import torch_npu

    active = loop // repeat - warmup
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=repeat, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )


def create_performance_exp_tensor() -> torch.Tensor:
    return torch.ones(10000, 10000).to(torch.float16).npu()


def run_performance_exp(iter_idx: int, tensor: torch.Tensor) -> None:
    if iter_idx != 1:
        return
    for _ in range(100):
        torch.exp(tensor)


def apply_stream_limit() -> None:
    if not is_full_core:
        torch.npu.set_stream_limit(torch.npu.current_stream(), aic_core_num, aiv_core_num)
    print(f"当前单算子模式仅使用：{aic_core_num} aic核，{aiv_core_num} aiv核")


def _init_cached_dispatch_handle(
    buffer: Any,
    x_npu: Feature,
    topk_idx_npu: torch.Tensor,
    topk_weights_npu: Optional[torch.Tensor],
    init_topk_weights_npu: Optional[torch.Tensor],
    init_cached_handle: bool,
) -> tuple[Any, Optional[GoldenHandle]]:
    if not init_cached_handle:
        return None, None
    init_topk_weights_npu = init_topk_weights_npu if init_topk_weights_npu is not None else topk_weights_npu
    if init_topk_weights_npu is None:
        raise ValueError("cached dispatch 初始化 handle 的第一次 dispatch 要求 init_topk_weights 非 None")
    _, _, _, ep_handle = buffer.dispatch(
        x_npu,
        topk_idx=topk_idx_npu,
        topk_weights=init_topk_weights_npu,
        num_experts=num_experts,
        expert_alignment=expert_alignment,
        do_cpu_sync=False,
    )
    return ep_handle, _actual_handle_to_golden(ep_handle)


def run_dispatch_npu(
    queue,
    rank: int,
    x: Feature,
    topk_idx: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    cached_handle: Optional[GoldenHandle],
    init_topk_weights: Optional[torch.Tensor] = None,
    init_cached_handle: bool = False,
) -> None:
    try:
        import torch_npu

        set_device(rank)
        ep_group = init_hccl_comm(rank)
        buffer = _create_buffer(ep_group)

        x_npu = _feature_to_npu(x)
        topk_idx_npu = topk_idx.npu()
        topk_weights_npu = None if topk_weights is None else topk_weights.npu()
        init_topk_weights_npu = None if init_topk_weights is None else init_topk_weights.npu()
        ep_handle = _golden_handle_to_ep_handle(cached_handle) if cached_handle is not None else None
        apply_stream_limit()
        cached_input_handle = None

        if init_cached_handle:
            ep_handle, cached_input_handle = _init_cached_dispatch_handle(
                buffer,
                x_npu,
                topk_idx_npu,
                topk_weights_npu,
                init_topk_weights_npu,
                init_cached_handle,
            )

        def _run_once():
            if ep_handle is not None:
                return buffer.dispatch(x_npu, topk_weights=topk_weights_npu, handle=ep_handle, do_cpu_sync=False)
            return buffer.dispatch(
                x_npu,
                topk_idx=topk_idx_npu,
                topk_weights=topk_weights_npu,
                num_experts=num_experts,
                expert_alignment=expert_alignment,
                do_cpu_sync=dispatch_do_cpu_sync_value(),
            )

        if not is_performance:
            recv_x, recv_topk_idx, recv_topk_weights, handle = _run_once()
        else:
            perf_exp_tensor = create_performance_exp_tensor()
            with get_profiling(str(dispatch_profiling_path)) as prof:
                for i in range(loop):
                    recv_x, recv_topk_idx, recv_topk_weights, handle = _run_once()
                    run_performance_exp(i, perf_exp_tensor)
                    torch_npu.npu.synchronize()
                    prof.step()

        torch_npu.npu.synchronize()
        dist.destroy_process_group()
        queue.put([
            rank,
            {
                "recv_x": _feature_to_cpu(recv_x),
                "recv_topk_idx": None if recv_topk_idx is None else recv_topk_idx.detach().cpu(),
                "recv_topk_weights": None if recv_topk_weights is None else recv_topk_weights.detach().cpu(),
                "handle": _actual_handle_to_golden(handle),
                "cached_input_handle": cached_input_handle,
            },
        ])
    except Exception:
        print("============================ Dispatch Exception ============================")
        print(f"Rank {rank} failed:\n{traceback.format_exc()}")
        queue.put([rank, None])


def run_combine_npu(
    queue,
    rank: int,
    combine_x: Feature,
    handle: GoldenHandle,
    combine_topk_weights: Optional[torch.Tensor],
) -> None:
    try:
        import torch_npu

        set_device(rank)
        ep_group = init_hccl_comm(rank)
        buffer = _create_buffer(ep_group)

        combine_x_npu = _feature_to_npu(combine_x)
        ep_handle = _golden_handle_to_ep_handle(handle)
        topk_weights_npu = None if combine_topk_weights is None else combine_topk_weights.npu()
        apply_stream_limit()

        def _run_once():
            return buffer.combine(combine_x_npu, handle=ep_handle, topk_weights=topk_weights_npu)

        if not is_performance:
            combined_x, combined_topk_weights = _run_once()
        else:
            combined_x, combined_topk_weights = _run_once()
            perf_exp_tensor = create_performance_exp_tensor()
            perf_sync_tensor = feature_main(combine_x_npu).new_ones(())
            with get_profiling(str(combine_profiling_path)) as prof:
                for i in range(loop):
                    perf_sync_tensor.fill_(1)
                    dist.all_reduce(perf_sync_tensor, group=ep_group)
                    combined_x, combined_topk_weights = _run_once()
                    run_performance_exp(i, perf_exp_tensor)
                    torch_npu.npu.synchronize()
                    prof.step()

        torch_npu.npu.synchronize()
        timing_output = feature_main(combined_x).clone()
        extract_combine_kernel_timing(timing_output, aiv_core_num, rank)
        dist.destroy_process_group()
        queue.put([
            rank,
            {
                "combined_x": _feature_to_cpu(combined_x),
                "combined_topk_weights": (
                    None if combined_topk_weights is None else combined_topk_weights.detach().cpu()
                ),
            },
        ])
    except Exception:
        print("============================ Combine Exception ============================")
        print(f"Rank {rank} failed:\n{traceback.format_exc()}")
        queue.put([rank, None])


def run_continue_npu(
    queue,
    rank: int,
    x: Feature,
    topk_idx: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    init_topk_weights: Optional[torch.Tensor] = None,
    init_cached_handle: bool = False,
) -> None:
    try:
        import torch_npu

        set_device(rank)
        ep_group = init_hccl_comm(rank)
        buffer = _create_buffer(ep_group)

        x_npu = _feature_to_npu(x)
        topk_idx_npu = topk_idx.npu()
        topk_weights_npu = None if topk_weights is None else topk_weights.npu()
        init_topk_weights_npu = None if init_topk_weights is None else init_topk_weights.npu()
        apply_stream_limit()
        ep_handle, cached_input_handle = _init_cached_dispatch_handle(
            buffer,
            x_npu,
            topk_idx_npu,
            topk_weights_npu,
            init_topk_weights_npu,
            init_cached_handle,
        )

        def _run_once():
            if ep_handle is not None:
                recv_x, recv_topk_idx, recv_topk_weights, handle = buffer.dispatch(
                    x_npu,
                    topk_weights=topk_weights_npu,
                    handle=ep_handle,
                    do_cpu_sync=False,
                )
            else:
                recv_x, recv_topk_idx, recv_topk_weights, handle = buffer.dispatch(
                    x_npu,
                    topk_idx=topk_idx_npu,
                    topk_weights=topk_weights_npu,
                    num_experts=num_experts,
                    expert_alignment=expert_alignment,
                    do_cpu_sync=dispatch_do_cpu_sync_value(),
                )
            recv_x_main, recv_scales = _split_recv_x_and_scales(recv_x)
            combine_source_x = recv_x if is_continue_use_dispatch_output else recv_x_main
            combine_x = _make_combine_x(combine_source_x)
            combine_topk = (
                gen_combine_topk_weight_cpu(rank, feature_main(combine_x).shape[0]).npu()
                if combine_with_topk_weights
                else None
            )
            combined_x, combined_topk_weights = buffer.combine(
                combine_x,
                handle=handle,
                topk_weights=combine_topk,
            )
            return recv_x, recv_scales, recv_topk_idx, recv_topk_weights, handle, combine_x, combine_topk, combined_x, combined_topk_weights

        if not is_performance:
            recv_x, recv_scales, recv_topk_idx, recv_topk_weights, handle, combine_x, combine_topk, combined_x, combined_topk_weights = _run_once()
        else:
            perf_exp_tensor = create_performance_exp_tensor()
            with get_profiling(str(continue_profiling_path)) as prof:
                for i in range(loop):
                    recv_x, recv_scales, recv_topk_idx, recv_topk_weights, handle, combine_x, combine_topk, combined_x, combined_topk_weights = _run_once()
                    run_performance_exp(i, perf_exp_tensor)
                    torch_npu.npu.synchronize()
                    prof.step()

        torch_npu.npu.synchronize()
        timing_output = feature_main(combined_x).clone()
        extract_combine_kernel_timing(timing_output, aiv_core_num, rank)
        dist.destroy_process_group()
        queue.put([
            rank,
            {
                "recv_x": _feature_to_cpu(recv_x),
                "recv_scales": None if recv_scales is None else recv_scales.detach().cpu(),
                "recv_topk_idx": None if recv_topk_idx is None else recv_topk_idx.detach().cpu(),
                "recv_topk_weights": None if recv_topk_weights is None else recv_topk_weights.detach().cpu(),
                "handle": _actual_handle_to_golden(handle),
                "cached_input_handle": cached_input_handle,
                "combine_x": _feature_to_cpu(combine_x),
                "combine_topk_weights": None if combine_topk is None else combine_topk.detach().cpu(),
                "combined_x": _feature_to_cpu(combined_x),
                "combined_topk_weights": (
                    None if combined_topk_weights is None else combined_topk_weights.detach().cpu()
                ),
            },
        ])
    except Exception:
        print("============================ Continue Exception ============================")
        print(f"Rank {rank} failed:\n{traceback.format_exc()}")
        queue.put([rank, None])


def gen_npu(target_func, **server_kwargs):
    if server_num > 1:
        print(f"multi_server scene!!! {server_num=}, {world_size=}, {server_index=}")
        rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    else:
        print("single_server scene!!!")
        rank_list = list(range(world_size))
    print(f"rank list is: {rank_list}")

    mp.set_start_method("forkserver", force=True)
    with Manager() as manager:
        result_queue = manager.Queue()
        proc_list = []
        for rank in rank_list:
            kwargs = {"queue": result_queue, "rank": rank}
            for key, value_list in server_kwargs.items():
                kwargs[key] = value_list[rank]
            proc = Process(target=target_func, kwargs=kwargs)
            print("============================== Start Proc ==============================")
            proc.start()
            proc_list.append(proc)

        rank_outputs = [None] * rank_per_dev
        for _ in proc_list:
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output
        for proc in proc_list:
            proc.join()

    if None in rank_outputs:
        print("[ERROR] Task failed! Please check subprocess logs.")
        raise RuntimeError("NPU 子进程失败")
    return rank_outputs


# endregion


# ======================================================================
# region 结果对比
# ======================================================================

def build_compare_config() -> CompareConfig:
    return CompareConfig(
        server_index=server_index,
        rank_per_dev=rank_per_dev,
        server_ranks=server_ranks,
        dispatch_with_handle=dispatch_with_handle,
        dispatch_with_topk_weights=dispatch_uses_topk_weights(),
        check_invalid_padding=check_invalid_padding,
        strict_dispatch_metadata_order=strict_dispatch_metadata_order,
        print_detail_max_elements=compare_print_detail_max_elements,
    )


def compare_dispatch_outputs(
    rank_outputs: List[Dict[str, Any]],
    case_data: Dict[str, Any],
    tag: str = "dispatch",
) -> Dict[str, bool]:
    return compare_dispatch_outputs_impl(rank_outputs, case_data, build_compare_config(), tag)


def compare_combine_outputs(rank_outputs: List[Dict[str, Any]], expected_goldens: List[Any], tag: str) -> Dict[str, bool]:
    config = build_compare_config()
    expected_host = expected_goldens[config.server_ranks]
    timing_bytes = aiv_core_num * 16 * 2 * 8
    for local_rank_id, output in enumerate(rank_outputs):
        actual_main = feature_main(output["combined_x"])
        expected_main = feature_main(expected_host[local_rank_id].combined_x)
        if actual_main.numel() * actual_main.element_size() < timing_bytes:
            continue
        timing_elements = timing_bytes // actual_main.element_size()
        actual_main.view(-1)[:timing_elements].copy_(expected_main.view(-1)[:timing_elements])
    print(f"[TIMING] {tag}: excluded the first {timing_bytes} bytes of combined_x from accuracy comparison")
    return compare_combine_outputs_impl(rank_outputs, expected_goldens, tag, config)


def print_summary(tag: str, result: Dict[str, bool]) -> None:
    print_summary_impl(tag, result)


# endregion


# ======================================================================
# region 主流程
# ======================================================================

def _dispatch_topk_weights_for_npu_run(case_data: Dict[str, Any]) -> List[Optional[torch.Tensor]]:
    return dispatch_topk_weights_for_npu(case_data)


def _dispatch_init_topk_weights_for_npu_run(case_data: Dict[str, Any]) -> List[Optional[torch.Tensor]]:
    if dispatch_with_handle:
        return case_data["topk_weights_per_rank"]
    return [None for _ in range(world_size)]


def _dispatch_compare_case_data(case_data: Dict[str, Any], rank_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dispatch_with_handle:
        return case_data
    cached_handles = [
        output["cached_input_handle"] if output.get("cached_input_handle") is not None else output["handle"]
        for output in rank_outputs
    ]
    compare_case_data = dict(case_data)
    compare_case_data["dispatch_goldens"] = [
        replace(dispatch_golden, handle=cached_handle)
        for dispatch_golden, cached_handle in zip(case_data["dispatch_goldens"], cached_handles)
    ]
    return compare_case_data


def _run_dispatch_test(case_data: Dict[str, Any]) -> Dict[str, bool]:
    clean_previous_profiling_result([dispatch_profiling_path])
    rank_outputs = gen_npu(
        run_dispatch_npu,
        x=case_data["x_per_rank"],
        topk_idx=case_data["topk_idx_per_rank"],
        topk_weights=_dispatch_topk_weights_for_npu_run(case_data),
        cached_handle=[None for _ in range(world_size)],
        init_topk_weights=_dispatch_init_topk_weights_for_npu_run(case_data),
        init_cached_handle=[dispatch_with_handle for _ in range(world_size)],
    )
    compare_case_data = _dispatch_compare_case_data(case_data, rank_outputs)
    result = compare_dispatch_outputs(rank_outputs, compare_case_data, "dispatch")
    if is_performance:
        print_performance_result("dispatch", dispatch_profiling_path, dispatch_op_types)
    return result


def _run_combine_test(case_data: Dict[str, Any]) -> Dict[str, bool]:
    combine_x_per_rank = combine_x_for_dispatch_goldens(case_data["dispatch_goldens"])
    combine_topk_weights = combine_topk_weights_for_npu(case_data["dispatch_goldens"])
    clean_previous_profiling_result([combine_profiling_path])
    rank_outputs = gen_npu(
        run_combine_npu,
        combine_x=combine_x_per_rank,
        handle=[d.handle for d in case_data["dispatch_goldens"]],
        combine_topk_weights=combine_topk_weights,
    )
    result = compare_combine_outputs(rank_outputs, case_data["combine_goldens"], "combine")
    if is_performance:
        print_performance_result("combine", combine_profiling_path, combine_op_types)
    return result


def _run_continue_test(case_data: Dict[str, Any]) -> Dict[str, bool]:
    clean_previous_profiling_result([continue_profiling_path])
    rank_outputs = gen_npu(
        run_continue_npu,
        x=case_data["x_per_rank"],
        topk_idx=case_data["topk_idx_per_rank"],
        topk_weights=_dispatch_topk_weights_for_npu_run(case_data),
        init_topk_weights=_dispatch_init_topk_weights_for_npu_run(case_data),
        init_cached_handle=[dispatch_with_handle for _ in range(world_size)],
    )
    compare_case_data = _dispatch_compare_case_data(case_data, rank_outputs)
    result = compare_dispatch_outputs(rank_outputs, compare_case_data, "continue.dispatch")

    actual_combine_x = [output["combine_x"] for output in rank_outputs]
    actual_handles = [output["handle"] for output in rank_outputs]
    actual_topk_weights = [output["combine_topk_weights"] for output in rank_outputs] if combine_with_topk_weights else None
    continue_combine_golden = gen_combine_golden_cpu(
        actual_combine_x,
        actual_handles,
        topk_weights_per_rank=actual_topk_weights,
    )
    result.update(compare_combine_outputs(rank_outputs, continue_combine_golden, "continue.combine"))
    if is_performance:
        print_performance_result("continue", continue_profiling_path, continue_op_types)
    return result


if __name__ == "__main__":
    print_setting()
    if is_check_setting:
        check_setting()

    case_data = generate_case_cpu()
    print("Generate golden success.")

    if is_cpu_only:
        print("[INFO] is_cpu_only=True，仅生成/校验 golden cache，不执行 NPU 算子。")
        raise SystemExit(0)

    result_dispatch = None
    result_combine = None
    result_continue = None

    if is_dispatch:
        result_dispatch = _run_dispatch_test(case_data)
    if is_combine:
        result_combine = _run_combine_test(case_data)
    if is_continue:
        result_continue = _run_continue_test(case_data)

    if result_dispatch is not None:
        print_summary("dispatch", result_dispatch)
        save_accuracy_result("dispatch", result_dispatch)
    if result_combine is not None:
        print_summary("combine", result_combine)
        save_accuracy_result("combine", result_combine)
    if result_continue is not None:
        print_summary("continue", result_continue)
        save_accuracy_result("continue", result_continue)

# endregion
